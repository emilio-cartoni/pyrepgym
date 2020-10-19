import numpy as np
import pyrepgym
import gym
from pyrepgym.envs.grid import make_grid

'''
    Demo showing random actions
    
    each position is above the object.
    each destination position is random.

'''
env = gym.make("PyRepEnv-v0", render_mode='human')
env.reset()
# show a grid of points delimiting the space of action in x and y

if False:
    handles, poses = make_grid(env.action_space['macro_action'])
else:
    from gym import spaces
    xmin = -0.35 -0.6
    xmax = -0.10 -0.6
    ymin = -0.19 -0.3
    ymax = 0.39 -0.3

    asp = spaces.Box(
                              low=np.array([[xmin, ymin], [xmin, ymin]]),
                              high=np.array([[xmax, ymax], [xmax, ymax]]),
                              dtype=float)
    handles, poses = make_grid(asp)

pos = env.objects['cube'].get_position()
for t in range(1):
    pos = env.objects['cube'].get_position()
    env.step_joints(np.zeros(7))
    action = env.action_space.sample()
    action["macro_action"][1,:] = pos[:2]
    env.step(action)

#env.close()

