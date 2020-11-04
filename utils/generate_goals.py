import numpy as np
import matplotlib.pyplot as plt
import pyrepgym
import gym
from pyrepgym.envs.grid import make_grid
from real_robots.envs import Goal

env = gym.make("PyRepEnv-v0", render_mode='console')
env.reset()
goals = []
for n in range(50):
    print("Generating Goal {}".format(n))

    action = env.action_space.sample()
    action['render'] = True
    obj_pos = action["macro_action"][0]
    env.objects['cube'].set_position(np.hstack([obj_pos, env.cube_pos[2]]))
    env.objects['cube'].set_quaternion(quaternion=[0, 0, 0, 1])
    action["macro_action"] = None
    obs, _, _, _ = env.step(action)
    retina1 = obs['retina']
    initial_pose = env.objects['cube'].get_pose()

    action = env.action_space.sample()
    action['render'] = True
    obj_pos = action["macro_action"][0]
    env.objects['cube'].set_position(np.hstack([obj_pos, env.cube_pos[2]]))
    env.objects['cube'].set_quaternion(quaternion=[0, 0, 0, 1])
    action["macro_action"] = None
    obs, _, _, _ = env.step(action)
    retina2 = obs['retina']
    final_pose = env.objects['cube'].get_pose()


    goal = Goal()
    goal.challenge = '2D'
    goal.subtype = str(1)
    goal.initial_state = {'cube' : initial_pose}
    goal.final_state = {'cube' : final_pose}
    goal.retina_before = retina1
    goal.retina = retina2
    goal.mask = None

    goals.append(goal)

env.close()

np.savez('allGoalsPyRep.npy', goals)

for n, g in enumerate(goals):
    plt.figure()
    plt.title("PRE {}".format(n))
    plt.imshow(g.retina_before)
    plt.figure()
    plt.title("POST {}".format(n))
    plt.imshow(g.retina)
plt.show()


all_i = np.vstack([g.initial_state['cube'] for g in goals])
all_f = np.vstack([g.final_state['cube'] for g in goals])
all_g = np.vstack([all_i, all_f])
plt.scatter(all_g[:,0], all_g[:,1])
plt.show()
