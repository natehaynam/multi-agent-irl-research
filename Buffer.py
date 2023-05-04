import numpy as np
import torch

class Buffer:
    """Replay buffer for agents with identical action spaces
    who observe entire state and finish (done) simultaneously"""

    def __init__(self, capacity, dim_info, device):
        self.capacity = capacity
        self.agent_ids = list(dim_info.keys())
        obs_dim = dim_info[self.agent_ids[0]][0]
        self.act_dim = dim_info[self.agent_ids[0]][1]
        self.obs = np.zeros((capacity, *obs_dim))
        self.next_obs = np.zeros((capacity, *obs_dim))
        self.done = np.zeros(capacity, dtype=bool)
        self.actions = {ids:np.zeros((capacity, self.act_dim))
            for ids in self.agent_ids}
        self.rewards = {ids:np.zeros(capacity)
            for ids in self.agent_ids}
        self._index = 0
        self._size = 0
        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs[self.agent_ids[0]]
        self.next_obs[self._index] = next_obs[self.agent_ids[0]]
        self.done[self._index] = done[self.agent_ids[0]]
        for agent_id in self.agent_ids:
            self.actions[agent_id][self._index] = action[agent_id]
            self.rewards[agent_id][self._index] = reward[agent_id]

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        # action and reward will be dictionaries by agent_id
        obs = torch.from_numpy(self.obs[indices]).float().to(self.device)
        next_obs = torch.from_numpy(self.next_obs[indices]).float().to(self.device)
        done = torch.from_numpy(self.done[indices]).float().to(self.device)
        action = {}
        reward = {}
        for agent_id in self.agent_ids:
            reward[agent_id] = torch.from_numpy(
                self.rewards[agent_id][indices]).float().to(self.device)
            action[agent_id] = torch.from_numpy(
                self.actions[agent_id][indices]).float().to(self.device)
        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size