import os
import shutil
import math
import genesis as gs
import numpy as np
import torch

from panda_env_reach import PandaReachGenesisEnv
from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity, RigidJoint
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 9,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
            "finger_joint1": 0.0,
            "finger_joint2": 0.0,
        },
        "dof_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        # termination
        "termination_if_distance_less_than": 0.06,
        "termination_if_angle_more_than": 70.0,
        # base pose
        "episode_length_s": 10.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        # "reward_scales": {"distance_penalty": 1.0, "action_rate": -0.005, "angle_penalty": 0.5},
        "reward_scales": {"distance_penalty": 1.0, "action_rate": -0.005},
    }
    return env_cfg


if __name__ == "__main__":
    # Создание и обучение
    exp_name = "Panda_1"
    log_dir = f"logs/{exp_name}"
    max_iterations = 1000
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    train_cfg = get_train_cfg("Panda_1", max_iterations)
    env_cfg = get_cfgs()
    num_obs = 24
    num_envs = 4096
    env = PandaReachGenesisEnv(env_cfg, num_obs, num_envs, render=False)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

# TODO: 1. Надо доделать функцию награды. 2. Надо запустить обучение робота на 10 средах сразу
