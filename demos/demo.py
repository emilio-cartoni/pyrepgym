import numpy as np
import pyrepgym
import gym


'''
    Demo showing random actions
    
    each position is above the object.
    each destination position is random.

'''
env = gym.make("PyRepEnv-v0", render_mode='human')
env.reset()

pos = env.objects['cube'].get_position()
for t in range(1):
    pos = env.objects['cube'].get_position()
    env.step_joints(np.zeros(7))
    action = env.action_space.sample()
    action["macro_action"][1,:] = pos[:2]
    env.step(action)

#env.close()

