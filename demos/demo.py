import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrepgym
import gym
from pyrepgym.envs.grid import make_grid

'''
    Demo showing random actions
    
    each position is above the object.
    each destination position is random.

'''

env = gym.make("PyRepEnv-v0")

# build the simulation 
env.render("human")
# reset positions
env.reset()
# show a grid of points delimiting the space of action in x and y
make_grid()

pos = env.objects[0].get_position()
for t in range(100):
    env.step_joints(np.zeros(7))

    action = pos[:2], env.action_space.sample()[0]
    env.step(action)
    pos = env.objects[0].get_position()

env.close()
