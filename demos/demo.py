import numpy as np
import pyrepgym
import gym
from pyrepgym.envs.grid import make_grid

'''
    Demo showing random actions
    
    each position is above the object.
    each destination position is random.

'''
print("START")
env = gym.make("PyRepEnvRos-v0", render_mode='console')
print("START2")
env.reset()
print("START3")
# show a grid of points delimiting the space of action in x and y
handles, poses = make_grid(env.action_space['macro_action'])

pos = env.objects['cube'].get_position()
for t in range(10):
    pos = env.objects['cube'].get_position()
    env.step_joints(np.zeros(7))
    action = env.action_space.sample()
    action["macro_action"][1,:] = pos[:2]
    env.step(action)

env.close()

