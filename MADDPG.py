import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from CNNAgent import Agent
from Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, device='cpu'):
        # create Agent(actor-critic) for each agent and replay buffer
        self.buffer = Buffer(capacity, dim_info, device)
        self.agents = {}
        for agent_id in dim_info.keys():
            self.agents[agent_id] = Agent(dim_info, actor_lr, critic_lr)
        self.dim_info = dim_info

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def add(self, obs, action, reward, next_obs, done):
        # NOTE that the experience is a dict with agent name as its key
        act = {}
        for agent_id in obs.keys():
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                act[agent_id] = np.eye(self.dim_info[agent_id][1])[a]
            else:
                act[agent_id] = action[agent_id]
        self.buffer.add(obs, act, reward, next_obs, done)

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffer)
        indices = np.random.choice(total_num, size=batch_size, replace=False)
        o, a, r, n_o, d = self.buffer.sample(indices)
        next_act = {}
        for agent_id in a.keys():
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)
        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        return o, a, r, n_o, d, next_act

    def select_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()
            a = self.agents[agent].action(o)  # torch.Size([1, act_dim])
            actions[agent] = a.squeeze(0).argmax().item()
            self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma):
        wandb_v = []
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            act_copy = {a:b for a,b in act.items()}
            # update critic
            critic_value = agent.critic_value(obs, act)
           # print("###################", agent_id, "#####################")
            #print("critic_value", critic_value)
            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(next_obs, next_act)
            target_value = reward[agent_id] + \
                gamma * next_target_critic_value * (1 - done)
            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            av_list = agent.update_critic(critic_loss)
            self.logger.info(f'{agent_id} critic gradient list: {av_list}')

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs, model_out=True)
            #temp_act = act[agent_id] # Seems to conform to original alg. Can be removed
            act_copy[agent_id] = action
            actor_loss = -agent.critic_value(obs, act_copy).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            av_list = agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            wandb_v.append([agent_id, critic_loss.item(), actor_loss.item(), torch.mean(critic_value), torch.mean(target_value)])
        return wandb_v

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def save(self, reward, episode=None):
        """save actor parameters of all agents and training reward to `res_dir`"""
        name = "model.pt" if episode is None else f"model_{episode}.pt"
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, name)
        )
        # with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
        #     pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance