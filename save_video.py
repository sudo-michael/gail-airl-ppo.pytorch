import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer

import gym


def run(args):
    env = make_env(args.env_id)
    env = gym.Monitor(env, './videos/test')


    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
    )


    done = False



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--rollout_length", type=int, default=50000)
    p.add_argument("--num_steps", type=int, default=10 ** 7)
    p.add_argument("--eval_interval", type=int, default=10 ** 5)
    p.add_argument("--env_id", type=str, default="Hopper-v3")
    p.add_argument("--algo", type=str, default="gail")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
