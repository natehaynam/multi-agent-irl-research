from pettingzoo.test import parallel_api_test
from pettingzoo.test import render_test
from gridWorld import CustomEnv

env = CustomEnv(render_mode = "human")
for i in range(10):
    obs = env.reset()
    done = False
    while env.agents:
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_space(agent).sample()
        obs, rewards, dones, truncated, infos = env.step(actions)

#parallel_api_test(env, num_cycles=1000)
#render_test(env)
