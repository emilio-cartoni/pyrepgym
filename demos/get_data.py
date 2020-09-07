import numpy as np
import matplotlib.pyplot as plt
from pyrepgym.envs.grid import make_grid

import pyrepgym
import gym

'''
Collect random postures 

the dataset is filtered by the desired space limits and used 
to regress the weights of a neural network for inverse kinematics.
'''

env = gym.make("PyRepEnv-v0")
env.render("console")
env.reset()
make_grid()

data = []
for k in range(20000):
    p = env.step_joints(np.zeros(7))
    p = env.step_explore()
    data.append(np.hstack(p))
    if k % 500 == 0:
        np.savetxt("data_explore", np.vstack(data))
        print(k)
data = np.vstack(data)

np.savetxt("data_explore", data)
env.close()
