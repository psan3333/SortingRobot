import genesis as gs
import numpy as np
import torch
import math
import logging

from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
from genesis.vis.camera import Camera

# from ultralytics import YOLO
from typing import List
from datetime import datetime
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
        cam_size=320,
        render=False,
        device="cuda",
    ):

        logging.basicConfig(
            level=logging.INFO, filename=f"text_logs/logs.log", filemode="w"
        )
        logging.info(f"Random seed: {torch.seed()}")
        self.device = torch.device(device)
        self.dt = 0.01  # сколько времени длится один кадр
        self.env_cfg = env_cfg
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.reward_scales = env_cfg["reward_scales"]
        self.grasp_detector = grasp_detector
        self.cam_size = cam_size
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]
        self.hover_distance = 0.4
        self.num_actions = env_cfg[
            "num_actions"
        ]  # agent doesn't control gripper's fingers -> 7
        self.gripper_fingers_angle_offset = 0.75

        # initialize Genesis
        gs.init(backend=gs.gpu, logging_level="warning")
        logging.info(
            f"{datetime.today().strftime('%Y-%m-%d_%H:%M:%S')} - Logging started!"
        )
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
        self.robot_cam: Camera = self.scene.add_camera(
            res=(self.cam_size, self.cam_size),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            GUI=False,
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
            lower=np.array([-87, -87, -87, -87, -12, -87, -87, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 87, 87, 100, 100]),
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
        self.grasp_angles = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)

        # robot data
        self.dofs_pos = torch.zeros_like(self.actions)
        self.dofs_vel = torch.zeros_like(self.actions)
        self.default_grasp = torch.tensor(
            [0.04, 0.04], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self.gripper_pos = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()
        self.default_dofs_pos = self.panda_arm.get_dofs_position(self.motor_dofs)[
            :, : self.num_actions
        ]

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
            (self.num_envs,), device=self.device, dtype=torch.int64
        )
        self.batch_norm = torch.func.vmap(
            torch.linalg.norm
        )  # parallel distance computation
        self.max_episode_length = math.ceil(10 / self.dt)  # 1000
        self.step_counter = 0
        self.grasp_frames_counter = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.int64
        )
        self.grasp_frame_cnt_limit = 30
        self.all_envs_idx = torch.arange(
            0, self.num_envs, device=self.device, dtype=torch.int64
        )
        self.extras = dict()

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.dofs_pos[envs_idx] = self.default_dofs_pos[envs_idx]
        self.dofs_vel[envs_idx] = 0.0
        new_dofs = torch.cat(
            [
                self.dofs_pos[envs_idx],
                torch.zeros((len(envs_idx), 2), device=self.device, dtype=gs.tc_float),
                self.default_grasp[envs_idx],
            ],
            dim=-1,
        )
        self.panda_arm.set_dofs_position(
            position=new_dofs,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.panda_arm.zero_all_dofs_velocity(envs_idx)
        num_envs_to_reset = len(envs_idx)
        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        pick_object_pos = torch.cat(
            [
                torch.rand(
                    (num_envs_to_reset, 1), device=self.device, dtype=gs.tc_float
                )
                * 0.1
                + 0.4,  # x
                torch.rand(
                    (num_envs_to_reset, 1), device=self.device, dtype=gs.tc_float
                )
                * 0.6
                - 0.3,  # y
                torch.tensor([0.05], device=self.device, dtype=gs.tc_float).repeat(
                    num_envs_to_reset, 1
                ),  # z
            ],
            dim=-1,
        )
        self.object.set_pos(pos=pick_object_pos, envs_idx=envs_idx)
        self.object_pos[envs_idx] = self.object.get_pos(envs_idx=envs_idx)
        self.default_object_pos[envs_idx] = self.object.get_pos(envs_idx=envs_idx)
        pick_object_pos[..., 2] = 0.025
        hover_targets = torch.clone(pick_object_pos)
        hover_targets[..., 2] = self.hover_distance
        lift_dest_targets = torch.cat(
            [
                torch.rand(
                    (num_envs_to_reset, 1), device=self.device, dtype=gs.tc_float
                )
                * 0.1
                + 0.3,  # x
                torch.rand(
                    (num_envs_to_reset, 1), device=self.device, dtype=gs.tc_float
                )
                * 0.6
                - 0.3,  # y
                torch.tensor([0.5], device=self.device, dtype=gs.tc_float).repeat(
                    num_envs_to_reset, 1
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
        self.extract_grap_angles(envs_idx)
        self.task_number[envs_idx] = 0
        self.grasp_frames_counter[envs_idx] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def set_cam_pos(self, env_idx):
        cam_pos: np.ndarray = (
            self.object_pos[env_idx].cpu().numpy() + self.scene.envs_offset[env_idx]
        )
        lookat = cam_pos.copy()
        cam_pos[2] = 0.8
        up = np.array([1.0, 0.0, 0.0])
        self.robot_cam.set_pose(pos=cam_pos, lookat=lookat, up=up)

    def extract_grap_angles(self, envs_idx):
        rgb = np.zeros(
            (len(envs_idx), self.cam_size, self.cam_size, 3), dtype=np.float32
        )
        depth = np.zeros(
            (len(envs_idx), self.cam_size, self.cam_size), dtype=np.float32
        )
        for i, env_idx in enumerate(envs_idx):
            self.set_cam_pos(env_idx)
            rgb[i], depth[i], _, _ = self.robot_cam.render(rgb=True, depth=True)

        depth = np.expand_dims(depth, axis=3)
        rgb, depth = (
            rgb.transpose((0, 3, 1, 2)),
            depth.transpose((0, 3, 1, 2)),
        )
        x = torch.from_numpy(np.concatenate((rgb, depth), axis=1))
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.grasp_detector.predict(xc)
            q_img, ang_img, width_img = post_process_output(
                pred["pos"], pred["cos"], pred["sin"], pred["width"]
            )
            for i, env_idx in enumerate(envs_idx):
                env_grasps: List[Grasp] = detect_grasps(
                    q_img[i], ang_img[i], width_img[i], 3
                )
                if len(env_grasps) > 0:
                    self.grasp_angles[env_idx] = float(env_grasps[0].angle)

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return

    def get_gripper_angles(self):
        get_angle_from_second_joint = self.dofs_pos[:, 3] < 0
        gripper_angle = self.dofs_pos[:, 1]
        gripper_angle[get_angle_from_second_joint] -= self.dofs_pos[
            get_angle_from_second_joint, 3
        ]
        gripper_fingers_angle = (
            self.dofs_pos[:, 0] + self.gripper_fingers_angle_offset + self.grasp_angles
        )
        return torch.cat(
            [gripper_angle.unsqueeze(1), gripper_fingers_angle.unsqueeze(1)], dim=-1
        )

    def get_dofs_for_grasp(self, envs_idx):
        zeros = torch.zeros((len(envs_idx), 2), device=self.device, dtype=gs.tc_float)
        dofs_pos = torch.cat(
            [self.dofs_pos[envs_idx], self.get_gripper_angles()[envs_idx], zeros],
            dim=-1,
        )
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
        gripper_angles = self.get_gripper_angles()
        target_dof_pos = torch.cat(
            [target_dof_pos, gripper_angles, self.default_grasp], dim=-1
        )
        target_dof_pos[self.task_number == 2, -2:] = 0.0
        grasp_task = self.grasp_frames_counter > 0
        grasp_envs_idx = grasp_task.nonzero(as_tuple=False).flatten()
        dof_pos_for_grasp = self.get_dofs_for_grasp(grasp_envs_idx)
        self.panda_arm.control_dofs_position(
            dof_pos_for_grasp,
            self.motor_dofs,
            envs_idx=grasp_envs_idx,
        )
        target_envs_idx = (
            torch.logical_not(grasp_task).nonzero(as_tuple=False).flatten()
        )
        self.panda_arm.control_dofs_position(
            target_dof_pos[target_envs_idx],
            self.motor_dofs,
            envs_idx=target_envs_idx,
        )
        self.scene.step()

        # update counter
        self.episode_length_buf += 1
        self.dofs_pos[:] = self.panda_arm.get_dofs_position(self.motor_dofs)[
            :, : self.num_actions
        ]
        self.dofs_vel[:] = self.panda_arm.get_dofs_velocity(self.motor_dofs)[
            :, : self.num_actions
        ]
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

        dist = self.batch_norm(
            self.target_pos[self.all_envs_idx, self.task_number].squeeze(dim=1)
            - self.gripper_pos
        )

        self.obs_buf = torch.cat(
            [
                dist.unsqueeze(dim=1),  # 1
                self.target_pos[self.all_envs_idx, self.task_number].squeeze(dim=1)
                - self.gripper_pos,  # 3
                self.target_pos[self.all_envs_idx, self.task_number].squeeze(
                    dim=1
                ),  # 3
                self.gripper_pos,  # 3
                self.dofs_pos,  # 5
                self.dofs_vel,  # 5
                self.actions,  # 5
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def check_for_reset(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        reached_the_target = torch.logical_and(
            self.batch_norm(
                self.target_pos[self.all_envs_idx, 2].squeeze(dim=1) - self.object_pos
            )
            < self.env_cfg["termination_if_distance_less_than"],
            self.task_number == 2,
        )
        reached_cnt = torch.sum(reached_the_target)
        if reached_cnt:
            logging.warning(f"The target was reached: {reached_cnt}")
        self.reset_buf |= reached_the_target

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_high_velocity_penalty(self):
        return torch.sum(torch.square(self.dofs_vel), dim=1)

    def _reward_dist_to_target_obj(self):
        # count distance to reachable object
        try:
            last_task_envs = self.task_number == 2
            distance_to_task = self.batch_norm(
                self.target_pos[self.all_envs_idx, self.task_number].squeeze(dim=1)
                - self.gripper_pos
            )
            distance_to_task[last_task_envs] = self.batch_norm(
                self.target_pos[last_task_envs, 2].squeeze(dim=1)
                - self.object_pos[last_task_envs]
            )
            charge_reward_envs = (
                distance_to_task < self.env_cfg["termination_if_distance_less_than"]
            )
            total_reward = -distance_to_task + 2.0
            total_reward *= self.task_number + 1
            total_reward[charge_reward_envs] += 5.0
            envs_for_grasping = torch.logical_and(
                charge_reward_envs, self.task_number == 1
            )
            self.grasp_frames_counter[
                torch.logical_or(envs_for_grasping, self.grasp_frames_counter > 0)
            ] += 1
            total_reward[
                torch.logical_or(envs_for_grasping, self.grasp_frames_counter > 0)
            ] = 3.0  # reward agent for achieving grasping step
            stop_grasping_envs = self.grasp_frames_counter >= self.grasp_frame_cnt_limit
            self.grasp_frames_counter[stop_grasping_envs] = 0
            total_reward[torch.logical_and(charge_reward_envs, last_task_envs)] += 15.0
            self.task_number[charge_reward_envs] += 1
            self.task_number[self.task_number > 2] = 2
            charging_reward = torch.sum(charge_reward_envs)
            second_task_started = torch.sum(self.task_number == 1)
            last_task_started = torch.sum(self.task_number == 2)
            if second_task_started and charging_reward:
                logging.info(
                    f"{datetime.today().strftime('%Y-%m-%d_%H:%M:%S')} - Hovered over target: {second_task_started}"
                )
            if last_task_started and charging_reward:
                logging.info(
                    f"{datetime.today().strftime('%Y-%m-%d_%H:%M:%S')} - Grabbing object: {last_task_started}"
                )
            return total_reward
        except Exception as e:
            print(e)
            raise e
        # change task number

    def close(self):
        return
