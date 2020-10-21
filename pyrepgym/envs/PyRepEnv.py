import numpy as np
import gym
from gym import spaces
from pyrepgym.envs.iknn import Iknn
from real_robots.envs import Goal
import os
import cv2

import rospy
import numpy as np
import std_msgs.msg
import sensor_msgs.msg
from ias_coppelia_sim_iiwas.ros import ROSCSControllerIiwasAirHockey
from ias_coppelia_sim_iiwas.sensors import RealSense

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape


IMAGE_TOPIC_NAME = 'kai/has/to/look/up/the/final/topic/name'
CAMERA_DELAY = 5

class AwesomeROSControllerIiwas(ROSCSControllerIiwasAirHockey):

    def __init__(self, headless=False, verbose=True, time_step=0.05, auto_start=False):
        ROSCSControllerIiwasAirHockey.__init__(self, headless, verbose, time_step, auto_start)
        self.last_image = self._t
        print("SELFT:", self._t)

    def _load_scene_components(self, **kwargs):
        ROSCSControllerIiwasAirHockey._load_scene_components(self, **kwargs)
        # load camera
        self.camera = RealSense.create(color=True, depth=False,
                                       position=[-0.9, 0.1, 1.1],
                                       orientation=[np.pi, 0., 0.])

        self.camera.set_handle_explicitly()

    def _start_ros_interface(self):
        ROSCSControllerIiwasAirHockey._start_ros_interface(self)
        # publisher for camera
        self.video_publisher = rospy.Publisher(IMAGE_TOPIC_NAME,
                                               sensor_msgs.msg.Image,
                                               queue_size=1)

    def _publish(self):
        ROSCSControllerIiwasAirHockey._publish(self)

        
        if (self._t - self.last_image) > (CAMERA_DELAY - 0.001):
            # publish image
            img = self.camera.capture_rgb()
            img_ = np.ascontiguousarray(img).tostring()

            header = std_msgs.msg.Header()
            header.stamp = rospy.Time(self._t)

            image_msg = sensor_msgs.msg.Image()
            image_msg.header = header
            image_msg.data = img_
            self.video_publisher.publish(image_msg)
            self.last_image = self._t
            print("camera taken", self._t)
        else:
            print(self._t)



class PyRepEnv	(gym.Env):
    ''' Custom PyRep Iiwas Environment that follows gym interface.
    '''

    metadata = {'render.modes': ['human', 'console']}

    intrinsic_timesteps = int(15e3)
    extrinsic_timesteps = int(10)

    def __init__(self, render_mode="console"):
        '''
            Args:
                render_mode: (string), 'human' for gui interactive simulation,
                'console' for offline simulation

            TODO: avoid qt dependency in console mode

        '''

        assert render_mode in PyRepEnv.metadata['render.modes']

        super(PyRepEnv, self).__init__()

        # Actions are defined as pairs of points on the 2D space
        # of the table (start point, destination point)
        self.macro_space = spaces.Box(
                                  low=np.array([[-1.20, -0.44], [-1.20, -0.44]]),
                                  high=np.array([[-0.90, 0.44], [-0.90,  0.44]]),
                                  dtype=float)

        self.action_space = spaces.Dict({
                            "macro_action": self.macro_space,
                            "render": spaces.MultiBinary(1)})

        obj_obs = {}
        used_objects = ['cube']
        for obj in used_objects:
            high = np.array([np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max])
            obj_obs[obj] = gym.spaces.Box(-high, high, dtype=float)

        self.observation_space = spaces.Dict({
            'retina': gym.spaces.Box(0, 255, [240, 320, 3], dtype=np.uint8),
            'mask' : gym.spaces.Box(0, 255, [240, 320, 3], dtype=np.int32),
            'object_positions': gym.spaces.Dict(obj_obs),
            'goal' : gym.spaces.Box(0, 255, [240, 320, 3], dtype=np.uint8),
            'goal_mask' : gym.spaces.Box(0, 255, [240, 320, 3], dtype=np.int32),
            'goal_positions' : gym.spaces.Dict(obj_obs)
            })

        self.no_retina = self.observation_space.spaces['retina'].sample()*0
        self.no_mask = self.observation_space.spaces['mask'].sample()*0
        self.goal = Goal(retina=self.observation_space.spaces['goal'].sample()*0)

        self.table_baseline=0.42
        self.table_above=0.6
        self.cube_on_table = 0.36902
        self.move_duration=np.array([3])
        self.ik=Iknn()
        self.objects={}

        ''' Start simulation
            Args:
                mode: (string), one of 'human' or 'console'
        '''
        self.headless=render_mode != "human"
        print("INIT", self.headless)
        rospy.init_node('coppelia_sim_iiwas_node')
        self.robot=AwesomeROSControllerIiwas(headless=False,#self.headless,
                                        auto_start=False, time_step=0.05)

        print("I2")
        self.robot.open_simulation()
        print("I3")
        self.robot.start_simulation()
        print("I4")

        rospy.Subscriber(IMAGE_TOPIC_NAME, sensor_msgs.msg.Image, self.receive_camera)

        self.timestep = 0
        print("I5")
        self.goals = None
        self.goal_idx = -1

    def receive_camera(self, image):
        print("Received camera image.")
        self.last_image = image

    def load_goals(self):
        self.goals = list(np.load(
                self.goals_dataset_path, allow_pickle=True).items())[0][1]

    def set_goals_dataset_path(self, path):
        assert os.path.exists(path), "Non existent path {}".format(path)
        self.goals_dataset_path = path

    def set_goal(self):
        if self.goals is None:
            self.load_goals()

        self.goal_idx += 1
        self.goal = self.goals[self.goal_idx]

        for obj in self.goal.initial_state.keys():
            pose = self.goal.initial_state[obj]
            self.objects[obj].set_pose(pose)

        for obj in self.goal.final_state.keys():
            self.goal.final_state[obj] = self.goal.final_state[obj][:3]

        return self.get_observation()

    def evaluateGoal(self):
        initial_state = self.goal.initial_state  # noqa F841
        final_state = self.goal.final_state
        score = 0
        for obj in final_state.keys():
            if obj not in self.objects:
                pass
            p = np.array(self.objects[obj].get_position())
            p_goal = np.array(final_state[obj][:3])
            pos_dist = np.linalg.norm(p_goal-p)
            # Score goes down to 0.25 within 10cm
            pos_const = -np.log(0.25) / 0.10
            pos_value = np.exp(- pos_const * pos_dist)
            objScore = pos_value
            # print("Object: {} Score: {:.4f}".format(obj,objScore))
            score += objScore

        # print("Goal score: {:.4f}".format(score))
        return self.goal.challenge, score

    def makeObject(self, color=[1, 0, 0], size=[0.05, 0.05, 0.05]):
        ''' Make a standard cuboid object

            Args:
                color: (list or array of 3 floats), RGB color
                size: (list or array of 3 floats), w, h, d sizes

            Returns:

                handle to pyrep object

        '''

        #pos=self.action_space.sample()['macro_action'][0]
        pos = [-1.0, -0.15]
        object_handle=Shape.create(
            mass=0.002,
            type=PrimitiveShape.CUBOID,
            color=color,
            size=size,
            position=[pos[0], pos[1], self.cube_on_table])
        object_handle.set_bullet_friction(10e9)
        self.objects['cube'] = object_handle

    def control_objects_limits(self):
        '''
        reset positions if an object goes out of the limits
        '''
        for obj in self.objects:
            x, y, z = self.objects[obj].get_position()
            if z < 0.35 or x > -0.9 or abs(y) > 0.44: #fallen off the table or too far
                self.objects[obj].set_pose(pose=[-1.0, -0.15, self.cube_on_table, 0, 0, 0, 1])


    def new_scene_conversion(self, pos):
        # This function converts from the coordinates of this scene
        # to the coordinates of the old scene (i.e. keeping the distance
        # from the robot the same). This function can be discarded
        # once the robot inverse kynematic is retrained for the new scene
        new_pos = pos.copy()
        new_pos[0] = new_pos[0] + 0.8272
        new_pos[1] = new_pos[1] + 0.2564
        return new_pos

    def move_to(self, arm, pos=None, joints=None):
        ''' Move gripper to next position, expressed in joint space or
            x,y,z coordinates.

            Args:
                arm: (string), one of 'LEFT_ARM' or 'RIGHT_ARM'
                pos: (array or list), x,y,z, position to reach
                joints: (array or list) the 7 angles in radiants describing
                    the posture to be reached
        '''
        if joints is not None:
            self.robot.goto_joint(
                arm,
                joints.reshape(1, -1),
                self.move_duration)
            self.robot.wait_for_goto(arm)
        elif pos is not None:
            pos = self.new_scene_conversion(pos)
            joints=self.ik.get_joints(pos)
            joints[6]=0.5*np.pi
            self.move_to(arm, joints=joints)

    def grasp(self, gripper, grasp_amp, torque):
        ''' Move gripper claws

            Args:
                gripper: (string), one of 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'
                grasp_amp: (float), amplitude of opening, from 0 (closed)
                    to 100 (open).
                torque: (float) amount of force, from 0 (none) to 100 (maximum)
        '''
        self.robot.grasp(gripper, grasp_amp, torque)
        self.robot.wait_for_grasp(gripper)

    def goHome(self, duration=3.0):
        self.robot.goto_joint('LEFT_ARM', np.zeros((1,7)), np.array([duration]))
        self.robot.wait_for_goto('LEFT_ARM')

    def render(self, mode="console"):
        # @TODO do we need to implement this?
        pass

    def reset(self):
        """ Restart simulation"""
        self.robot.stop_simulation()
        self.robot.start_simulation()
        self.timestep = 0
        self.goHome()
        self.grasp('LEFT_GRIPPER', 0, 100)
        self.makeObject()
        #Note: it seems when simulation is stopped, you need to recreate the camera
        self.cam = self.robot.camera # self.setupCamera()
        return self.get_observation()

    def get_observation(self, render=True):
        pos=self.robot.get_gripper_position('LEFT_ARM')

        if render:
            rgb = self.cam.capture_rgb()  # returns np.array height x width x 3
            rgb = cv2.resize(rgb*256, (320,240)).astype('uint8')
        else:
            rgb = self.no_retina.astype('uint8')

        cube_pos = self.objects['cube'].get_position()

        obj_pos = {'cube' : cube_pos}

        observation = {'position': pos,
                       'retina': rgb,
                       'mask' : self.no_mask,
                       'object_positions' : obj_pos,
                       'goal' : self.goal.retina,
                       'goal_mask' : self.goal.mask,
                       'goal_positions' : self.goal.final_state
                      }
        return observation

    def step(self, action):

        macro_action = action['macro_action']
        render = action['render']

        if macro_action is not None:
            print("Executing:", macro_action)
            start, dest=macro_action

            p1_up=np.hstack([start, self.table_above])
            p1_down=np.hstack([start, self.table_baseline])
            p1_up=np.hstack([start, self.table_above])
            p2_up=np.hstack([dest, self.table_above])
            p2_down=np.hstack([dest, self.table_baseline])

            ''' action is composed of several movements.
                    1 gripper moved above the start position, 
                    2 gripper moved down
                    3 gripper moved to the destination position
                    8 gripper moved up
                    5 go back home
            '''
            self.move_to(arm="LEFT_ARM", pos=p1_up)
            self.move_to(arm="LEFT_ARM", pos=p1_down)
            gpos = self.robot.get_gripper_position('LEFT_ARM')[0]
            self.move_to(arm="LEFT_ARM", pos=p2_down)
            self.move_to(arm="LEFT_ARM", pos=p2_up)
            self.goHome()

            self.control_objects_limits()

        else:
            gpos = None
            print("No action to execute, just observe.")

        observation = self.get_observation(render)
        observation['reached_pos'] = gpos # for debug purpose

        reward = 0
        done = False
        self.timestep += 1
        if self.goal_idx < 0:
            if self.timestep >= self.intrinsic_timesteps:
                done = True
        else:
            if self.timestep >= self.extrinsic_timesteps:
                done = True

        info = {}

        return observation, reward, done, info

    def close(self):
        ''' Close simulation '''
        self.robot.stop_simulation()
        self.robot.__del__()

    def step_joints(self, joints):
        ''' Change joint position -- Debug purposes '''

        self.robot.goto_joint(
            'LEFT_ARM',
            joints.reshape(1, -1),
            self.move_duration)
        self.robot.wait_for_goto('LEFT_ARM')
        pos=self.robot.get_gripper_position('LEFT_ARM')

        return pos

    def step_explore(self, verticalGripper=False):
        ''' Random joint posture -- Debug purpose '''

        joint_poses=np.array([[-20, 30, 0., -60, 0., 90, 0.]]) / 180 * np.pi
        joint_poses += (20*np.random.randn(7)*[1, 1, 0, 1, 0, 1, 0]) / 180 * np.pi

        if verticalGripper:
            joint_poses[0, 2] = 0
            joint_poses[0, 4] = 0
            joint_poses[0, 6] = 0
            joint_poses[0, 5] = np.pi - (joint_poses[0, 1] - joint_poses[0, 3])
            joint_poses[0, 6] = joint_poses[0, 0]

        self.robot.goto_joint('LEFT_ARM',
                              joint_poses,
                              self.move_duration)
        self.robot.wait_for_goto('LEFT_ARM')

        pos=self.robot.get_gripper_position('LEFT_ARM')
        return np.hstack((joint_poses.ravel(), np.hstack(pos)))

