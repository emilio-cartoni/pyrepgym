import numpy as np
import pyrepgym
import gym
from pyrepgym.envs.grid import make_grid
import itertools

'''
    Demo to check movements along all the edges of the action space
    plus the two main diagonals

'''

env = gym.make("PyRepEnv-v0", render_mode='human')
env.reset()
# show a grid of points delimiting the space of action in x and y
handles, poses = make_grid(env.action_space['macro_action'], points=2)
# change from -1.2 to -1.25
combinations = itertools.combinations(poses, 2)

for c in combinations:
    action = env.action_space.sample()
    action["macro_action"][0,:] = np.array(c[0])
    action["macro_action"][1,:] = np.array(c[1])
    print("I will execute {}".format(action))
    env.step(action)

env.close()

