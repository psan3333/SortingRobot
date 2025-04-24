import genesis as gs
import numpy as np
import torch
import math

from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity, RigidJoint
from genesis.vis.camera import Camera

# from ultralytics import YOLO
from typing import List
from genesis.vis.camera import Camera
from inference.models.grconvnet import GenerativeResnet
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.dataset_processing.grasp import Grasp


class PandaSort:
    def __init__(
        self,
        env_cfg,
        grasp_detector: GenerativeResnet,
        cam_size=224,
        render=False,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.dt = 0.01  # сколько времени длится один кадр
        self.env_cfg = env_cfg
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.reward_scales = env_cfg["reward_scales"]
        self.grasp_detector = grasp_detector
        self.cam_size = cam_size
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]

        # initialize Genesis
        gs.init(backend=gs.gpu, logging_level="warning")
        self.device = torch.device(device)
        self.scene: gs.Scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=int(0.5 / self.dt),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=render,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.panda_arm: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        )
        self.joint_names = env_cfg["dof_names"]
        self.object: RigidEntity = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0, 0.05))
        )

        # setup cameras for each environment
        self.robot_cams: List[Camera] = []
        for i in range(self.num_envs):
            self.robot_cams.append(
                self.scene.add_camera(
                    res=(self.cam_size, self.cam_size),
                    pos=(3.5, 0.0, 2.5),
                    lookat=(0, 0, 0.5),
                    fov=30,
                    GUI=False,
                )
            )
        self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))

        # set positional gains
        self.motor_dofs = [
            self.panda_arm.get_joint(name).dof_idx_local for name in self.joint_names
        ]
        self.panda_arm.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            dofs_idx_local=self.motor_dofs,
        )
        # set velocity gains
        self.panda_arm.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
            dofs_idx_local=self.motor_dofs,
        )
        self.panda_arm.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.motor_dofs,
        )

        # observations settings
        self.num_privileged_obs = None
        self.num_actions = 9  # robot has only 9 joints to control

        # prepare reward functions and set rewards scales depending on simulation FPS
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        # environment observation buffers
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.base_vector = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)

        # camera data - also environmnt
        self.rgb_frames = np.zeros(
            (self.num_envs, self.cam_size, self.cam_size, 3), dtype=np.float32
        )
        self.depth_frames = np.zeros(
            (self.num_envs, self.cam_size, self.cam_size), dtype=np.float32
        )
        self.grasps = torch.zeros(
            (self.num_envs, 5), device=self.device, dtype=gs.tc_float
        )
        self.on_ground_discount = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_float
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)

        # robot data
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.gripper_pos = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()
        self.default_dofs_pos = self.panda_arm.get_dofs_position(self.motor_dofs)

        # agent target data
        self.object_pos = self.object.get_pos()
        self.default_object_pos = self.object.get_pos()
        self.target_pos = torch.tensor(
            [0.5, 0.5, 0.5], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self.batch_norm = torch.func.vmap(
            torch.linalg.norm
        )  # parallel distance computation
        self.max_episode_length = math.ceil(10 / self.dt)  # 1000
        self.step_counter = 0
        self.extras = dict()

    def extract_grasps(self):
        envs = self.episode_length_buf.cpu().numpy() % 2 == 0
        depth = np.expand_dims(self.depth_frames[envs], axis=3)
        rgb, depth = (
            self.rgb_frames[envs].transpose((0, 3, 1, 2)),
            depth.transpose((0, 3, 1, 2)),
        )
        x = torch.from_numpy(np.concatenate((rgb, depth), 1))
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.grasp_detector.predict(xc)
            q_img, ang_img, width_img = post_process_output(
                pred["pos"], pred["cos"], pred["sin"], pred["width"]
            )
            idx = 0
            for env in range(self.num_envs):
                if envs[env]:
                    grasps: List[Grasp] = detect_grasps(
                        q_img[idx], ang_img[idx], width_img[idx], 1
                    )
                    idx += 1
                    if len(grasps) > 0:
                        grasp = grasps[0]
                        try:
                            data = torch.tensor(
                                [
                                    grasp.center[0],
                                    grasp.center[1],
                                    grasp.angle,
                                    grasp.length,
                                    grasp.width,
                                ],
                                device=self.device,
                                dtype=gs.tc_float,
                            )
                            data = data.to(self.device)
                            self.grasps[env] = data
                        except Exception:
                            pass

    def adjuct_camera_to_gripper(self):
        finger_pos = (
            self.panda_arm.get_joint(self.joint_names[-2]).get_pos().cpu().numpy()
        )
        joint_pos = (
            self.panda_arm.get_joint(self.joint_names[-3]).get_pos().cpu().numpy()
        )
        for i in range(self.num_envs):
            f_pos = finger_pos[i]
            j_pos = joint_pos[i]
            cam_pos = f_pos + 0.12 * (f_pos - j_pos) + self.scene.envs_offset[i]
            lookat = f_pos + 0.15 * (f_pos - j_pos) + self.scene.envs_offset[i]
            self.robot_cams[i].set_pose(pos=cam_pos, lookat=lookat)
            self.rgb_frames[i], self.depth_frames[i], _, _ = self.robot_cams[i].render(
                rgb=True, depth=True
            )

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def reset_idx(self, envs_idx):
        self.dof_pos[envs_idx] = self.default_dofs_pos[envs_idx]
        self.dof_vel[envs_idx] = 0.0
        self.panda_arm.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.panda_arm.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        new_object_pos = torch.cat(
            [
                torch.rand((len(envs_idx), 1), device=self.device, dtype=gs.tc_float)
                * 0.6
                + 0.1,  # x
                torch.rand((len(envs_idx), 1), device=self.device, dtype=gs.tc_float)
                * 0.6
                - 0.3,  # y
                torch.tensor([0.05], device=self.device, dtype=gs.tc_float).repeat(
                    len(envs_idx), 1
                ),  # z
            ],
            dim=-1,
        )
        self.object.set_pos(pos=new_object_pos, envs_idx=envs_idx)
        self.object_pos[envs_idx] = self.object.get_pos(envs_idx=envs_idx)
        self.default_object_pos[envs_idx] = self.object.get_pos(envs_idx=envs_idx)
        new_targets = torch.cat(
            [
                torch.rand((len(envs_idx), 1), device=self.device, dtype=gs.tc_float)
                * 0.5
                + 0.1,  # x
                torch.rand((len(envs_idx), 1), device=self.device, dtype=gs.tc_float)
                * 0.6
                - 0.3,  # y
                torch.tensor([0.5], device=self.device, dtype=gs.tc_float).repeat(
                    len(envs_idx), 1
                ),  # z
            ],
            dim=-1,
        )
        self.target_pos[envs_idx] = new_targets
        self.gripper_pos = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return

    def step(self, actions):
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            exec_actions * self.env_cfg["action_scale"] + self.default_dofs_pos
        )
        self.panda_arm.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()
        self.adjuct_camera_to_gripper()
        self.extract_grasps()

        # update counter
        self.episode_length_buf += 1
        self.dof_pos[:] = self.panda_arm.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.panda_arm.get_dofs_velocity(self.motor_dofs)
        self.gripper_pos[:] = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()
        self.check_for_reset()

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.obs_buf = torch.cat(
            [
                self.object_pos - self.gripper_pos,  # 3
                self.target_pos - self.object_pos,  # 3
                self.object_pos,  # 3
                self.target_pos,  # 3
                self.dof_pos,  # 9
                self.actions,  # 9
                self.grasps,  # 5
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def check_for_reset(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        reached_the_target = (
            self.batch_norm(self.target_pos - self.object_pos)
            < self.env_cfg["termination_if_distance_less_than"]
        )
        self.reset_buf |= reached_the_target

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dist_to_target_obj(self):
        # count distance to reacchable object
        distance = self.batch_norm(self.object_pos - self.gripper_pos)
        charge_reward = distance < self.env_cfg["termination_if_distance_less_than"]
        total_reward = -distance + 2.0
        total_reward[charge_reward] += 5.0

        # distance from object to target position
        target_pos_dist = self.batch_norm(self.target_pos - self.object_pos)
        objects_not_in_air = (
            torch.abs(self.default_object_pos[..., 2] - self.object_pos[..., 2]) < 0.05
        )
        objects_to_lift = torch.logical_and(charge_reward, objects_not_in_air)
        objects_in_air = torch.logical_and(
            charge_reward, torch.logical_not(objects_not_in_air)
        )
        if torch.sum(objects_to_lift):
            total_reward[objects_to_lift] -= target_pos_dist[objects_to_lift] * 2.0
        if torch.sum(objects_in_air):
            total_reward[objects_in_air] -= target_pos_dist[objects_in_air] * 0.5
        total_reward[
            target_pos_dist < self.env_cfg["termination_if_distance_less_than"]
        ] += 10.0
        return total_reward

    def close(self):
        return
