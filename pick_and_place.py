import genesis as gs
import numpy as np
import torch
import math

from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity, RigidJoint
from genesis.vis.camera import Camera
from genesis.utils.geom import transform_by_quat

# from ultralytics import YOLO
from typing import List
from inference.models.grconvnet import GenerativeResnet
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.dataset_processing.grasp import Grasp


# TODO: make object collections for different object types
# TODO: check grasp angle depending on robot base rotation
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
        self.hover_distance = 0.3
        self.num_actions = env_cfg[
            "num_actions"
        ]  # robot has only 9 joints to control, but fingers will be controled by algorithm

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
        self.grasp_angles = np.zeros((self.num_envs,), dtype=np.float32)
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

        # target pos:
        # 3 positions (x, y, z) to reach for each environment
        # target points: hover over object, grab object, move to the destination
        self.target_pos = torch.zeros(
            (self.num_envs, 3, 3), device=self.device, dtype=gs.tc_float
        )
        self.task_number = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.batch_norm = torch.func.vmap(
            torch.linalg.norm
        )  # parallel distance computation
        self.max_episode_length = math.ceil(10 / self.dt)  # 1000
        self.step_counter = 0
        self.grasp_frames_counter = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.int64
        )
        self.grasp_frame_cnt_limit = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.int64
        ).fill_(50)
        self.extras = dict()

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def reset_idx(self, envs_idx):
        self.dof_pos[envs_idx] = self.default_dofs_pos[envs_idx]
        self.dof_pos[envs_idx, -2:] = 0.04
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
        pick_object_pos = torch.cat(
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
        self.object.set_pos(pos=pick_object_pos, envs_idx=envs_idx)
        self.object_pos[envs_idx] = self.object.get_pos(envs_idx=envs_idx)
        self.default_object_pos[envs_idx] = self.object.get_pos(envs_idx=envs_idx)
        hover_targets: torch.Tensor = pick_object_pos.copy_()
        hover_targets[..., 2] = self.hover_distance
        lift_dest_targets = torch.cat(
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
        new_targets = torch.cat(
            [
                hover_targets.unsqueeze(dim=1),
                pick_object_pos.unsqueeze(dim=1),
                lift_dest_targets.unsqueeze(dim=1),
            ],
            dim=1,
        )  # shape^ (self.num_envs, 3, 3)
        self.target_pos[envs_idx] = new_targets
        self.gripper_pos = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()
        self.grasp_frames_counter[envs_idx] = 0
        # change camera positions
        for i in range(self.num_envs):
            if envs_idx[i]:
                # set default camera position above target object
                cam_pos = self.object_pos[i] + self.scene.envs_offset[i]
                cam_pos[2] = 0.8
                lookat = self.object_pos[i] + self.scene.envs_offset[i]
                up = np.array([1.0, 0.0, 0.0])
                self.robot_cams[i].set_pose(pos=cam_pos, lookat=lookat, up=up)

        self.extract_grap_angles(envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def extract_grap_angles(self, env_idx):
        rgb, depth, _, _ = self.robot_cams[env_idx].render(rgb=True, depth=True)
        depth = np.expand_dims(depth, axis=2)
        rgb, depth = (
            rgb.transpose((2, 0, 1)),
            depth.transpose((2, 0, 1)),
        )
        x = torch.from_numpy(
            np.expand_dims(np.concatenate((rgb, depth), axis=0), axis=0)
        )
        with torch.no_grad():
            ten_grasp_angles = []
            while len(ten_grasp_angles) < 10:
                xc = x.to(self.device)
                pred = self.grasp_detector.predict(xc)
                q_img, ang_img, width_img = post_process_output(
                    pred["pos"], pred["cos"], pred["sin"], pred["width"]
                )
                env_grasps: List[Grasp] = detect_grasps(q_img, ang_img, width_img, 3)
                ten_grasp_angles.append(env_grasps[0].angle)
            self.grasp_angles[env_idx] = np.array(ten_grasp_angles).mean()

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return

    def get_dofs_for_grasp(self, envs_idx):
        dofs_pos = self.dof_pos[envs_idx]
        dofs_pos[-2:] = 0.0
        return dofs_pos

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
        target_dof_pos = torch.cat([target_dof_pos, self.dof_pos[:, -2:]], dim=-1)
        grasp_task = self.grasp_frames_counter > 0
        dof_pos_for_grasp = self.get_dofs_for_grasp(grasp_task)
        self.panda_arm.control_dofs_position(
            dof_pos_for_grasp, self.motor_dofs, envs_idx=grasp_task
        )
        self.panda_arm.control_dofs_position(
            target_dof_pos, self.motor_dofs, envs_idx=torch.logical_not(grasp_task)
        )
        self.scene.step()

        # update counter
        self.episode_length_buf += 1
        self.dof_pos[:] = self.panda_arm.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.panda_arm.get_dofs_velocity(self.motor_dofs)
        self.gripper_pos[:] = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.check_for_reset()
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
                self.target_pos[:, self.task_number] - self.object_pos,  # 3
                self.object_pos,  # 3
                self.target_pos[:, self.task_number],  # 3
                self.dof_pos,  # 9
                self.actions,  # 9
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def check_for_reset(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        reached_the_target = torch.logical_and(
            self.batch_norm(self.target_pos[:, 2] - self.object_pos)
            < self.env_cfg["termination_if_distance_less_than"],
            self.task_number == 2,
        )
        self.reset_buf |= reached_the_target

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dist_to_target_obj(self):
        # count distance to reacchable object
        distance = self.batch_norm(
            self.target_pos[:, self.task_number] - self.gripper_pos
        )
        charge_reward = distance < self.env_cfg["termination_if_distance_less_than"]
        total_reward = -distance + 2.0
        total_reward[charge_reward] += 5.0
        grasping_step = np.logical_and(charge_reward, self.task_number == 1)
        self.grasp_frames_counter[
            torch.logical_or(grasping_step, self.grasp_frames_counter > 0)
        ] += 1
        stop_grasping = self.grasp_frames_counter >= self.grasp_frame_cnt_limit
        self.grasp_frames_counter[stop_grasping] = 0

        # change task number
        self.task_number[charge_reward] += 1

    def close(self):
        return
