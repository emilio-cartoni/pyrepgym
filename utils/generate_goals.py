import numpy as np
import matplotlib.pyplot as plt
from real_robots.envs import Goal
import rospy
from pyrepgym.envs.PyRepEnv import macro_space
import cv2

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from imageGenerator import AwesomeROSControllerIiwas

rospy.init_node('coppelia_sim_iiwas_node')

time_step = 0.05
real_time = True

cs_controller = AwesomeROSControllerIiwas(time_step=time_step)
cs_controller.open_simulation()
cs_controller.start_simulation()

cs_controller.goto_joint('LEFT_ARM', np.array([0,-0.20,0,-1,0,1.1,0]),
                            np.array([3]))
cs_controller.wait_for_goto('LEFT_ARM')
cs_controller.grasp('LEFT_GRIPPER',15, 100)
cs_controller.wait_for_grasp('LEFT_GRIPPER')

from gym import spaces

macro_space = spaces.Box(
                          low=np.array([[-1.10, -0.30], [-1.10, -0.30]]),
                          high=np.array([[-0.90, 0.30], [-0.90,  0.30]]),
                          dtype=float)


goals = []
for n in range(50):
    print("Generating Goal {}".format(n))

    obj_pos = macro_space.sample()[0]
    cs_controller.objects['cube'].set_position(np.hstack([obj_pos, 0.19]))
    cs_controller.objects['cube'].set_quaternion(quaternion=[0, 0, 0, 1])
    rgb = cs_controller.camera.capture_rgb()
    retina1 = cv2.resize(rgb*256, (320,240)).astype('uint8')
    initial_pose = cs_controller.objects['cube'].get_pose()


    obj_pos = macro_space.sample()[0]
    cs_controller.objects['cube'].set_position(np.hstack([obj_pos, 0.19]))
    cs_controller.objects['cube'].set_quaternion(quaternion=[0, 0, 0, 1])
    rgb = cs_controller.camera.capture_rgb()
    retina2 = cv2.resize(rgb*256, (320,240)).astype('uint8')
    final_pose = cs_controller.objects['cube'].get_pose()

    goal = Goal()
    goal.challenge = '2D'
    goal.subtype = str(1)
    goal.initial_state = {'cube' : initial_pose}
    goal.final_state = {'cube' : final_pose}
    goal.retina_before = retina1
    goal.retina = retina2
    goal.mask = None

    goals.append(goal)


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
