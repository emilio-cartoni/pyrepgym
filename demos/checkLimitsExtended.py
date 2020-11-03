import numpy as np
import pyrepgym
import gym
from pyrepgym.envs.grid import make_grid
import itertools
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

'''
    Demo to check movements along all the edges of the action space
    plus the two main diagonals

'''

env = gym.make("PyRepEnv-v0", render_mode='human')
env.reset()


def createPoint(pos, color):
    type = PrimitiveShape.CUBOID
    w, h, d = 0.01, 0.01, 0.01
    z = 0.5   

    object_handle = Shape.create(type=type, color=color,
                                 size=[w, h, d], static=True,
                                 respondable=False,
                                 position=[pos[0], pos[1], z])
    return object_handle

box_action_space = env.action_space['macro_action']
low_x = box_action_space.low[0][0]
high_x =  box_action_space.high[0][0]
low_y = box_action_space.low[0][1]
high_y =  box_action_space.high[0][1]

x_steps = int(np.ceil((high_x - low_x) / 0.025))
y_steps = int(np.ceil((high_y - low_y) / 0.025))

xs = np.linspace(low_x, high_x, x_steps)
ys = np.linspace(low_y, high_y, y_steps)
poses = [p for p in itertools.product(xs, ys)]

for pos in poses:
    pos = np.array(list(pos) + [env.table_baseline])
    joints = env.ik.get_joints(pos)
    env.robot.goto_joint('LEFT_ARM', joints, np.array([1]))
    env.robot.wait_for_goto('LEFT_ARM')
    x, o = env.robot.get_gripper_position('LEFT_GRIPPER')
    print("Desired pos:", pos)
    print("Reached pos:", x)
    print("Orientation:", o)
    if np.linalg.norm(pos-x) > 0.05:
        createPoint(pos, (1,0,0))
    else:
        createPoint(pos, (0,1,0))





