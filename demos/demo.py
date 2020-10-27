import numpy as np
import pyrepgym
import gym


'''
    Demo showing random actions

'''
SL = False  # set to True when using the real robots
env = gym.make("PyRepEnv-v0", render_mode='human', SL=SL)
env.reset()

for t in range(5):
    action = env.action_space.sample()
    print(action['macro_action'][0, :])
    print(action['macro_action'][1, :])
    action['macro_action'][1, :] = env.objects['cube'][:2]
    env.step(action)

env.close()

