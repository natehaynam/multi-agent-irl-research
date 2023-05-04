from gridWorld import CustomEnv
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from MADDPG import MADDPG
from main import get_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode-num', type=int, default=10, help='total episode num during evaluation')
    parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')
    parser.add_argument('--model', type=str, default=1, help='model num to run')

    args = parser.parse_args()

    model_dir = './results/7/' 
    env, dim_info = get_env(args.episode_length)
    env = CustomEnv(15,15,[5,5,5],render_mode = "human")
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, 'model_'+args.model+'.pt'))


    for episode in range(args.episode_num):
        obs = env.reset()
        done = False
        while env.agents:  # interact with the env for an episode
            flat_states = {agent: obs[agent].reshape(-1) for agent in obs.keys()}
            for agent in env.agents:
                actions = maddpg.select_action(flat_states)
            obs, rewards, dones, truncated, infos = env.step(actions)
        env.close()
