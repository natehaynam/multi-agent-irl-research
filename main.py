import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from gridWorld import CustomEnv
from MADDPG import MADDPG
import torch
import wandb
from PIL import Image

def get_env(ep_len=40, grid_length=15, grid_width=15, brp_list=[5,5,5]):
    """create environment and get observation and action dimension of each
    agent in this environment"""
    new_env = CustomEnv(grid_length, grid_width, brp_list, horizon=ep_len)
    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape)
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)
    return new_env, _dim_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_num', type=int, default=1000000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=40, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=50,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=2e4,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=1e-3, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e5), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=64, help='batch-size of replay buffer')
    parser.add_argument('--grid_length', type=int, default=15, help='Length of environment grid')
    parser.add_argument('--grid_width', type=int, default=15, help='Width of environment grid')
    parser.add_argument('--num_br', type=int, default=5, help='Number of blue and red gems in grid')
    parser.add_argument('--num_purple', type=int, default=5, help='Number of purple gems in grid')
    parser.add_argument('--actor_lr', type=float, default=2e-4, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=2e-4, help='learning rate of critic')
    args = parser.parse_args()

    # create folder to save result
    env_dir = './results'
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(args.episode_length, args.grid_length,
                            args.grid_width, [args.num_br]*2+[args.num_purple])
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size,
                    args.actor_lr, args.critic_lr, result_dir)

    wandb.init(
        project="maddpg_values",
        config={
        "buffer_capacity": int(args.buffer_capacity),
        "episodes": int(args.episode_num),
        }
    )
    wandb_v = []
    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    for episode in range(args.episode_num):
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs)

            next_obs, reward, done, truncated, info = env.step(action)
            maddpg.add(obs, action, reward, next_obs, done)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                wandb_v = maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        sum_reward, rewards = 0, []

        for agent_id, r in agent_reward.items():  # record reward
            sum_reward += r
            rewards.append(r)
    
        if(wandb_v):
            wandb.log({"sum_reward": sum_reward, "blue_agent": int(rewards[0]), "red_agent": int(rewards[1]), 
                    "blue_critic_loss": wandb_v[0][1], "blue_actor_loss": wandb_v[0][2], "blue_critic_value": wandb_v[0][3], "blue_target_value": wandb_v[0][4],
                    "red_critic_loss": wandb_v[1][1], "red_actor_loss": wandb_v[1][2], "red_critic_value": wandb_v[1][3], "red_target_value": wandb_v[1][4]})

        if (episode + 1) % 10 == 0:
            maddpg.save(episode_rewards, episode = episode + 1)
            img = Image.fromarray(env.render(mode='rgb_array'))
            img.save("env"+str(episode)+".jpg")

    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))