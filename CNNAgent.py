from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam

class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, dim_info, actor_lr, critic_lr):
        """dim_info is a list with observation shape and action length"""
        self.agent_ids = list(dim_info.keys())
        obs_shape = dim_info[self.agent_ids[0]][0]
        act_size = dim_info[self.agent_ids[0]][1]
        self.actor = ActorCNNetwork(obs_shape, act_size, hidden_dim=64)
        self.critic = CriticCNNetwork(obs_shape, act_size, hidden_dim=64)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])
        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        action = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])
        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach() # TODO refactor w above method

    def critic_value(self, obs, actions):
        acts = torch.cat(tuple([actions[id] for id in self.agent_ids]), 1)
        return self.critic(obs, acts).squeeze(1)

    def target_critic_value(self, obs, actions):
        acts = torch.cat(tuple([actions[id] for id in self.agent_ids]), 1)
        return self.target_critic(obs, acts).squeeze(1)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

class ActorCNNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, non_linear=nn.LeakyReLU):
        super(ActorCNNetwork, self).__init__()
        self.length, self.width, num_feats = obs_dim
        self.conv = nn.Sequential(
            nn.Conv2d(num_feats, 10, kernel_size=3, padding=1),
            non_linear(),
            nn.Conv2d(10, 20, kernel_size=3),
            non_linear(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 20, kernel_size=3),
            non_linear(),
        )

        self.lin = nn.Sequential(
            nn.Linear(320, hidden_dim),
            non_linear(),
            nn.Linear(hidden_dim, hidden_dim),
            non_linear(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = torch.permute(x, (0, 3, 1, 2))
        batch = x.shape[0]
        x = self.conv(x)
        x = x.reshape(batch, -1)
        out = self.lin(x)
        return out

class CriticCNNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, non_linear=nn.LeakyReLU):
        super(CriticCNNetwork, self).__init__()
        self.length, self.width, num_feats = obs_dim
        self.conv = nn.Sequential(
            nn.Conv2d(num_feats, 10, kernel_size=3, padding=1),
            non_linear(),
            nn.Conv2d(10, 20, kernel_size=3),
            non_linear(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 20, kernel_size=3),
            non_linear(),
        )

        self.lin = nn.Sequential(
            nn.Linear(330, hidden_dim),
            non_linear(),
            nn.Linear(hidden_dim, hidden_dim),
            non_linear(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, acts):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = torch.permute(x, (0, 3, 1, 2))
        batch = x.shape[0]
        x = self.conv(x)
        x = x.reshape(batch, -1)
        y = torch.cat((x, acts), axis=1)
        out = self.lin(y)
        return out