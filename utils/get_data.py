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

env = gym.make("PyRepEnv-v0", render_mode="console")
env.reset()
make_grid()
seed = np.random.randint(1000000)
datafile = "data_explore_{}".format(seed)
data = []
for k in range(5000):
    p = env.step_joints(np.zeros(7))
    p = env.step_explore(True)
    data.append(np.hstack(p))
    if k % 500 == 0:
        np.savetxt(datafile, np.vstack(data))
        print(k)
data = np.vstack(data)
np.savetxt(datafile, data)
env.close()
