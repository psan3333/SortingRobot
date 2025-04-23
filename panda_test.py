import argparse
import os
import pickle

import torch
from panda_env_reach import PandaReachGenesisEnv
from panda_train import get_cfgs, get_train_cfg
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    max_iterations = 300
    exp_name = "Panda_1"
    log_dir = f"logs/{exp_name}"
    train_cfg = get_train_cfg(exp_name, max_iterations)
    env_cfg = get_cfgs()
    num_obs = 24
    num_envs = 1
    env = PandaReachGenesisEnv(env_cfg, num_obs, num_envs, render=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
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
