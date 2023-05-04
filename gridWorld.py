import functools
import numpy as np
from gymnasium.spaces import Discrete, MultiBinary
import pygame
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

class CustomEnv(ParallelEnv):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    color_dict = {
            2: (0,0,255),
            3: (255,0,0),
            4: (153,50,204)
        }

    def __init__(self, length=15, width=15, num_brp=[10, 10, 10],
                 purple_reward_ratio=3, horizon=40, render_mode = None):
        super().__init__()
        self.steps = 0
        self.window_size = 600
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.num_blue, self.num_red, self.num_purple = tuple(num_brp)
        self.purple_reward_ratio = purple_reward_ratio
        self.length = length
        self.width = width
        self.horizon = horizon
        self.possible_agents = ["blue", "red"]
        self.agents = ["blue", "red"]
        self.bloc = np.array([0,0], dtype=np.int32)
        self.rloc = np.array([0,0], dtype=np.int32)

        # Layers, in order, are blue agent, red agent, then blue, red, purple gem
        self.obs_space = MultiBinary([length, width, 5])
        self.state = None
        self.act_space = Discrete(5)

        #mapping from action to direction
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0])
        }
        self.obs = np.zeros(self.obs_space.shape,
            dtype = self.obs_space.dtype)
        total = self.num_blue + self.num_red + self.num_purple
        indices = np.random.choice(self.length*self.width, total+2, replace=False)
        self.x = [11, 6, 2, 5, 11, 2, 4, 2, 10, 13, 6, 8, 1, 9, 6, 12, 7]

        self.y = [2, 12, 9, 10, 1, 3, 4, 12, 3, 4, 8, 8, 5, 10, 4, 10, 10]
        print(self.x)
        print(self.y)

    def reset(self, seed=0, options=None):
        x_i = self.x
        y_i = self.y
        # Blue gems
        self.obs[:,:,2][x_i[:self.num_blue],
            y_i[:self.num_blue]] = 1
        # Red gems
        self.obs[:,:,3][x_i[self.num_blue:self.num_blue+self.num_red],
            y_i[self.num_blue:self.num_blue+self.num_red]] = 1
        # Purple gems
        self.obs[:,:,4][x_i[self.num_blue+self.num_red:-2],
            y_i[self.num_blue+self.num_red:-2]] = 1
        # Blue agent
        self.obs[:,:,0][x_i[-2], y_i[-2]] = 1
        # Red agent
        self.obs[:,:,1][x_i[-1], y_i[-1]] = 1

        self.bloc[0], self.bloc[1] = x_i[-2], y_i[-2]
        self.rloc[0], self.rloc[1] = x_i[-1], y_i[-1]

        self.state = self.obs
        self.steps = 0
        self.agents = self.possible_agents

        if self.render_mode == "human":
            self.render()

        return {"blue": self.obs, "red": self.obs}

    def step(self, actions):

        # update blue agent loc
        blue_action = self._action_to_direction[actions["blue"]]
        self.state[self.bloc[0],self.bloc[1], 0] = 0
        self.bloc[0] = np.clip(self.bloc[0] + blue_action[0], 0, self.length - 1)
        self.bloc[1] = np.clip(self.bloc[1] + blue_action[1], 0, self.width - 1)
        self.state[self.bloc[0],self.bloc[1], 0] = 1
        # update red agent loc
        red_action = self._action_to_direction[actions["red"]]
        self.state[self.rloc[0],self.rloc[1], 1] = 0
        self.rloc[0] = np.clip(self.rloc[0] + red_action[0], 0, self.length - 1)
        self.rloc[1] = np.clip(self.rloc[1] + red_action[1], 0, self.width - 1)
        self.state[self.rloc[0],self.rloc[1], 1] = 1

        # calculate blue reward
        if self.state[self.bloc[0],self.bloc[1], 2] == 1:
            blue_reward = 1
            self.state[self.bloc[0],self.bloc[1], 2] = 0
        else:
            blue_reward = 0
        # calculate red reward
        if self.state[self.rloc[0],self.rloc[1], 3] == 1:
            red_reward = 1
            self.state[self.rloc[0],self.rloc[1], 3] = 0
        else:
            red_reward = 0
        # Calculate purple reward
        if self.state[self.bloc[0],self.bloc[1], 4] == 1 and \
            self.state[self.rloc[0],self.rloc[1], 4] == 1:
            blue_reward = red_reward = self.purple_reward_ratio
            self.state[self.bloc[0],self.bloc[1], 4] = 0
            self.state[self.rloc[0],self.rloc[1], 4] = 0

        if self.render_mode == "human":
            self.render()
        self.steps += 1
        #print(self.bloc, self.rloc)

        # check if there are no red, blue, purple gems on the board
        done = np.sum(self.state[:,:,2:]) == 0 or self.steps >= self.horizon
        color_to_r = {"blue": blue_reward, "red": red_reward}
        # Create outputs

        obs, rewards, dones, infos, truncated = {}, {}, {}, {}, {}
        for agent in self.agents:
            obs[agent] = self.state
            rewards[agent] = color_to_r[agent]
            dones[agent] = done
            truncated[agent] = done
            infos[agent] = {}

        if done:
            self.agents = []

        return obs, rewards, dones, truncated, infos

    def render(self, mode="human"):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.length*10, self.width*10))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.length*10, self.width*10))
        canvas.fill((255, 255, 255))
        pix_square_size = 10  # The size of a single grid square in pixels

        #draw the agents
        pygame.draw.rect(canvas, (0, 0, 255), pygame.Rect(
            pix_square_size * self.bloc, (pix_square_size, pix_square_size)))
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(
            pix_square_size * self.rloc, (pix_square_size, pix_square_size)))

        # draw blue, red, purple gems
        for k in range(2,5):
            for i in range(self.length):
                for j in range(self.width):
                    if self.state[i,j,k] == 1:
                        pygame.draw.circle(
                            canvas, self.color_dict[k],
                            (pix_square_size * (i+0.5), pix_square_size *(j+0.5)),
                            pix_square_size/2)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        return self.obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id):
        return self.act_space