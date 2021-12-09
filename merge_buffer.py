import os
import argparse
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.utils import collect_demo
from gail_airl_ppo.buffer import SerializedBuffer, Buffer


from tqdm import tqdm
import gym

def run(args):
    env = make_env(args.env_id)

    buffer_exp_1 = SerializedBuffer(
        path=args.b1, device=torch.device("cuda" if args.cuda else "cpu")
    )
    
    buffer_exp_2 = SerializedBuffer(
        path=args.b2, device=torch.device("cuda" if args.cuda else "cpu")
    )

    # half = buffer_exp_1.buffer_size // 2
    quater = buffer_exp_1.buffer_size // 4
    other = buffer_exp_1.buffer_size - quater

    # b1_states, b1_actions, b1_rewards, b1_dones, b1_next_states = buffer_exp_1.sample(half)
    # b2_states, b2_actions, b2_rewards, b2_dones, b2_next_states = buffer_exp_2.sample(half)

    b1_states, b1_actions, b1_rewards, b1_dones, b1_next_states = buffer_exp_1.sample(quater)
    b2_states, b2_actions, b2_rewards, b2_dones, b2_next_states = buffer_exp_2.sample(other)
    states = torch.cat((b1_states, b2_states), dim=0)
    actions = torch.cat((b1_actions, b2_actions), dim=0)
    rewards = torch.cat((b1_rewards, b2_rewards), dim=0)
    dones = torch.cat((b1_dones, b2_dones), dim=0)
    next_states = torch.cat((b1_next_states, b2_next_states), dim=0)


    buffer = Buffer(buffer_exp_1.buffer_size, env.observation_space.shape, env.action_space.shape, device=torch.device("cuda" if args.cuda else "cpu"))
    buffer.states = states
    buffer.actions = actions 
    buffer.rewards = rewards 
    buffer.dones = dones 
    buffer.next_states = next_states 

    buffer.save(
        os.path.join(
            "buffers",
            args.env_id,
            f"{args.b1}_{args.b2}_{args.name}.pth",
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--b1", type=str, required=True)
    p.add_argument("--b2", type=str, required=True)
    p.add_argument("--env_id", type=str, default="Hopper-v3")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()
    run(args)
