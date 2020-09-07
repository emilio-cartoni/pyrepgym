from vrep_iiwas.base import CSConntrollerIiwas
import numpy as np
import gym
from gym import spaces
from iknn import Iknn

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape


class PyRepEnv(gym.Env):
    ''' Custom PyRep Iiwas Environment that follows gym interface.
    '''

    metadata = {'render.modes': ['human', 'console']}

    def __init__(self, render_mode="human"):

        assert render_mode in PyRepEnv.metadata['render.modes']

        super(PyRepEnv, self).__init__()

        # Actions are defined as pairs of points on the 2D space
        # of the table (start point, destination point)
        center = [-0.1, 0.1]
        side = 0.15
        self.action_space = spaces.Box(
            # Lower limits for start and destination
            low=np.array([[-1, -1], [-1, -1]])*side + center,
            # Upper limits for start and destination
            high=np.array([[1, 1], [1, 1]])*side + center,
            dtype=float)

        self.table_baseline = 0.42
        self.table_above = 0.6
        self.move_duration = np.array([3])
        self.ik = Iknn()
        self.objects = []

    def makeObject(self, color=[1, 0, 0], size=[0.05, 0.05, 0.05]):

        pos = self.action_space.sample()[0]
        object_handle = Shape.create(
            mass=0.002,
            type=PrimitiveShape.CUBOID,
            color=color,
            size=size,
            position=[pos[0], pos[1], self.table_baseline])
        object_handle.set_bullet_friction(10e9)
        self.objects.append(object_handle)

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
            joints = self.ik.get_joints(pos)
            joints[6] = 0.5*np.pi
            self.move_to(arm, joints=joints)

    def grasp(self, gripper, grasp_amp, torque):
        self.robot.grasp(gripper,grasp_amp, torque)
        self.robot.wait_for_grasp(gripper)


    def render(self, mode="console"):
        self.headless = mode != "human"
        self.robot = CSConntrollerIiwas(headless=mode != "human",
                                        auto_start=False)

        self.robot.open_simulation()
        self.robot.start_simulation()

    def reset(self):
        """ """
        self.robot.stop_simulation()
        self.robot.start_simulation()

        self.makeObject()

    def step(self, action):

        start, dest = action

        p1_up = np.hstack([start, self.table_above])
        p1_down = np.hstack([start, self.table_baseline])
        p1_up = np.hstack([start, self.table_above])
        p2_up = np.hstack([dest, self.table_above])
        p2_down = np.hstack([dest, self.table_baseline])

        def move(joints):
            joints[6] = 0.5*np.pi
            self.robot.goto_joint(
                'LEFT_ARM',
                joints.reshape(1, -1),
                self.move_duration)
            self.robot.wait_for_goto('LEFT_ARM')

        self.robot.grasp('LEFT_GRIPPER', 60, 15)
        self.robot.wait_for_grasp('LEFT_GRIPPER')
        joints = self.ik.get_joints(p1_up)
        move(joints)
        joints = self.ik.get_joints(p1_down)
        move(joints)
        self.robot.grasp('LEFT_GRIPPER', 20, 10)
        self.robot.wait_for_grasp('LEFT_GRIPPER')
        self.robot.grasp('LEFT_GRIPPER', 20, 100)
        self.robot.wait_for_grasp('LEFT_GRIPPER')
        joints = self.ik.get_joints(p1_up)
        move(joints)
        joints = self.ik.get_joints(p2_up)
        move(joints)
        joints = self.ik.get_joints(p2_down)
        move(joints)
        self.robot.grasp('LEFT_GRIPPER', 60, 15)
        self.robot.wait_for_grasp('LEFT_GRIPPER')
        joints = self.ik.get_joints(p2_up)
        move(joints)

        pos = self.robot.get_gripper_position('LEFT_ARM')

    def step_joints(self, joints):
        self.robot.goto_joint(
            'LEFT_ARM',
            joints.reshape(1, -1),
            self.move_duration)
        self.robot.wait_for_goto('LEFT_ARM')
        pos = self.robot.get_gripper_position('LEFT_ARM')

        return pos

    def step_explore(self):

        joint_poses = np.array([[-20, 30, 0., -60, 0., 90, 0.]]) / 180 * np.pi
        joint_poses += (20*np.random.randn(7)*[1, 1, 0, 1, 0, 1, 0]) \
            / 180 * np.pi

        self.robot.goto_joint('LEFT_ARM',
                              joint_poses,
                              self.move_duration)
        self.robot.wait_for_goto('LEFT_ARM')

        pos = self.robot.get_gripper_position('LEFT_ARM')
        return np.hstack((joint_poses.ravel(), np.hstack(pos)))

    def close(self):
        self.robot.stop_simulation()
        self.robot.__del__()
