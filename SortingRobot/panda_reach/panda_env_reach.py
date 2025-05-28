import math
import genesis as gs
import numpy as np
import torch

from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity, RigidJoint
from genesis.utils.geom import quat_to_xyz


# TODO: переделать среду обучения агента без gym.Env - целиком на pytorch и genesis.
# За подсказками можно смотреть в go2_train.py в папке locomotion
class PandaReachGenesisEnv:
    def __init__(
        self,
        env_cfg,
        num_obs,
        num_envs=10,
        render=False,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.dt = 0.01  # сколько времени длится один кадр
        self.env_cfg = env_cfg
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.reward_scales = env_cfg["reward_scales"]

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
        self.panda_arm: RigidEntity = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.joint_names = env_cfg["dof_names"]
        self.object: RigidEntity = self.scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0, 0.05)))
        self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))

        ############ Optional: set control gains ############
        # set positional gains
        self.motor_dofs = [self.panda_arm.get_joint(name).dof_idx_local for name in self.joint_names]
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
        self.num_privileged_obs = None
        self.num_actions = 9
        self.num_obs = num_obs

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.base_vector = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.gripper_pos = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()
        self.object_pos = self.object.get_pos()
        self.angles = torch.zeros((self.num_envs), device=self.device, dtype=gs.tc_float)
        self.default_dofs_pos = self.panda_arm.get_dofs_position(self.motor_dofs)
        self.max_episode_length = math.ceil(10 / self.dt)  # 10 секунд по 50 кадров
        self.batch_norm = torch.func.vmap(torch.linalg.norm)
        self.step_counter = 0
        self.extras = dict()

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
        new_object_pos = 0.3 * torch.rand((len(envs_idx), 2), device=self.device, dtype=gs.tc_float) + 0.3
        new_object_pos = torch.cat(
            [new_object_pos, torch.tensor([0.05], device=self.device, dtype=gs.tc_float).repeat(len(envs_idx), 1)],
            dim=-1,
        )
        self.object.set_pos(pos=new_object_pos, envs_idx=envs_idx)
        self.object_pos[envs_idx] = new_object_pos
        self.gripper_pos = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dofs_pos
        self.panda_arm.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update counter
        self.episode_length_buf += 1
        self.dof_pos[:] = self.panda_arm.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.panda_arm.get_dofs_velocity(self.motor_dofs)
        self.gripper_pos[:] = self.panda_arm.get_joint(self.joint_names[-1]).get_pos()
        # self.angles[:] = self._angle_between_target()
        # check termination and reset
        # TODO: define termination factors
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        reached_the_target = (
            self.batch_norm(self.object_pos - self.gripper_pos) < self.env_cfg["termination_if_distance_less_than"]
        )
        self.reset_buf |= reached_the_target
        # self.reset_buf |= self.angles > self.env_cfg["termination_if_angle_more_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        self.rew_buf[reached_the_target] += 10.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.obs_buf = torch.cat(
            [
                self.object_pos - self.gripper_pos,
                # self.angles / 90.0,
                self.object_pos,
                self.dof_pos,
                self.actions,
            ],
            axis=-1,
        )
        # print(self.angles[:10])

        self.last_actions[:] = self.actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _calc_target_angle(self):
        base_joint: RigidJoint = self.panda_arm.get_joint(self.joint_names[0])
        base_pos = base_joint.get_pos()
        vector_to_object = self.object_pos - base_pos
        batched_func = torch.func.vmap(
            lambda u, v: (
                (
                    torch.acos(
                        torch.clip(
                            torch.dot(u[:2], v[:2]) / (torch.linalg.norm(u[:2]) * torch.linalg.norm(v[:2])), -1.0, 1.0
                        )
                    )
                )
                * 180
                / torch.pi
            )
        )
        self.target_angles = batched_func(vector_to_object, self.base_vector)
        self.target_angles = self.target_angles.unsqueeze(-1)

    def _angle_between_target(self):
        base_joint: RigidJoint = self.panda_arm.get_joint(self.joint_names[0])
        base_pos = base_joint.get_pos()
        base_quat = base_joint.get_quat()
        hand_angles = torch.abs(quat_to_xyz(base_quat)[..., 2])
        vector_to_object = torch.cat(
            [
                self.object_pos[:2] - base_pos[:2],
                torch.tensor([0.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1),
            ],
            dim=-1,
        )
        batched_func = torch.func.vmap(lambda u, v: (torch.atan2(torch.cross(u, v), torch.dot(u, v))) * 180 / torch.pi)
        target_angle = batched_func(vector_to_object, self.base_vector)
        angles = target_angle - hand_angles
        return angles.unsqueeze(-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_distance_penalty(self):
        distance = self.batch_norm(self.object_pos - self.gripper_pos)
        return -distance + 2.0

    # def _reward_angle_penalty(self):
    # return -torch.abs(self.angles) / 90.0

    def close(self):
        return
