import argparse
import os
import pickle

import torch
from pick_and_place import PandaSort
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import shutil
from datetime import datetime
from inference.models.grconvnet import GenerativeResnet


def get_train_cfg(exp_name):

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
            "max_iterations": 20001,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 1000,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 5,
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
        "termination_if_distance_less_than": 0.05,
        # base pose
        "episode_length_s": 10.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "num_obs": 25,
        "num_envs": 4,
        "reward_scales": {
            "dist_to_target_obj": 1.0,
            "high_velocity_penalty": -0.01,
            "action_rate": -0.005,
        },
    }
    return env_cfg


# TODO: изменить количество контролируемых звеньев робота, чтобы он точно обучался заадче этой
def main():

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    exp_name = f"PandaSort_{datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}_test"
    log_dir = f"logs/{exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    train_cfg = get_train_cfg(exp_name)
    env_cfg = get_cfgs()
    grasp_detector: GenerativeResnet = torch.load("./grasp_weights", weights_only=False)
    env = PandaSort(env_cfg, grasp_detector=grasp_detector, render=True)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = "./agent.pt"
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
