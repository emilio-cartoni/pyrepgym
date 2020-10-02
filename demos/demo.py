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

env = gym.make("PyRepEnv-v0", render_mode='console')
# reset positions
env.reset()
# show a grid of points delimiting the space of action in x and y
handles, poses = make_grid()


#Move right arm out of the way
rarm = env.robot.arms['RIGHT_ARM']
rarm.set_position(np.array([  5.6802,   5.2488,   0.1060]))

dati = []
if False:
    pos = env.objects[0].get_position()
    for t in range(1):
        pos = env.objects[0].get_position()
        env.step_joints(np.zeros(7))
        action = env.action_space.sample()
        action["macro_action"][:2] = pos[:2]
        env.step(action)
else:
    for pos in poses:
        action = env.action_space.sample()
        action["macro_action"][:2] = pos[:2]
        obs, _, _, _ = env.step(action)
        dati.append(obs['reached_pos'])
print(dati)
env.close()

a_dati = np.vstack(dati)
a_poses = np.vstack(poses)
a_poses = np.hstack([poses, np.ones((len(poses),1))*0.42])

np.save('reachData.npy', np.hstack([a_dati, a_poses]))


if False:
    import numpy as np
    data = np.load('reachData.npy')
    a_dati = data[:, :3]
    a_poses = data[:,3:]
    err = np.linalg.norm(a_dati - a_poses, axis=1).reshape(56,1)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(a_poses[:,0], a_poses[:,1], err, linewidth=0, antialiased=False)
    plt.show()

if False:
    cube.set_position(np.array([-0.05,0.39,0.37]))
    env.step_joints(np.zeros((1,7)))
    print(":", cube.get_position())
    pos = env.objects[0].get_position()
    env.step_joints(np.zeros(7))
    action = env.action_space.sample()
    action["macro_action"][:2] = pos[:2]
    env.step(action)


