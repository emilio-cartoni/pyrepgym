import rospy
import numpy as np
import std_msgs.msg
import sensor_msgs.msg
from ias_coppelia_sim_iiwas.base import CSControllerIiwasAirHokey
from ias_coppelia_sim_iiwas.sensors import RealSense
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrepgym.envs.grid import make_grid
from pyrepgym.envs.PyRepEnv import macro_space
import time
import tf2_ros
import geometry_msgs.msg

IMAGE_TOPIC_NAME = 'kai/has/to/look/up/the/final/topic/name'
CAMERA_DELAY = 5

class AwesomeROSControllerIiwas(CSControllerIiwasAirHokey):

    def __init__(self, headless=True, verbose=True, time_step=0.05, auto_start=False):
        CSControllerIiwasAirHokey.__init__(self, headless, verbose, time_step, auto_start)
        self.last_image = -10000
        self.cube_pos =[-1.0, -0.15, 0.191]
        self.objects = {}

        # publisher for camera
        self.video_publisher = rospy.Publisher(IMAGE_TOPIC_NAME,
                                               sensor_msgs.msg.Image,
                                               queue_size=1)

        self.tfBuffer = tf2_ros.Buffer()
        self.objpos_listener = tf2_ros.TransformListener(self.tfBuffer)

    def _load_scene_components(self, **kwargs):
        CSControllerIiwasAirHokey._load_scene_components(self, **kwargs)
        # load camera
        self.camera = RealSense.create(color=True, depth=False,
                                       position=[-0.9, 0., 1.1],
                                       orientation=[np.pi, 0., np.pi/2])

        self.camera.set_handle_explicitly()
        self.camera.set_model_collidable(False)
        # create object
        self.makeObject()
        #handles, poses = make_grid(macro_space)


    def publish(self):
        # set object position
        msg = self.tfBuffer.lookup_transform('coppelia_origin', 'orange', rospy.Time())
        x = msg.transform.translation.x
        y = msg.transform.translation.y
        z = msg.transform.translation.z
        rx = msg.transform.rotation.x
        ry = msg.transform.rotation.y
        rz = msg.transform.rotation.z
        rw = msg.transform.rotation.w

        self.set_cube_pose([x, y, z, rx, ry, rz, rw])
        print("Cube set to:", [x, y, z, rx, ry, rz, rw])

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
        print("Resolution:",img.shape)

    def makeObject(self, color=[1, 0, 0], size=[0.05, 0.05, 0.05]):
        ''' Make a standard cuboid object
            Args:
                color: (list or array of 3 floats), RGB color
                size: (list or array of 3 floats), w, h, d sizes
            Returns:
                handle to pyrep object
        '''

        #pos=self.action_space.sample()['macro_action'][0]
        object_handle=Shape.create(
            mass=0.002,
            type=PrimitiveShape.CUBOID,
            color=color,
            size=size,
            position=self.cube_pos)
        object_handle.set_bullet_friction(10e9)
        self.objects['cube'] = object_handle

    def set_cube_pose(self, obj_pose):
        self.objects['cube'].set_pose(pose=obj_pose)



if __name__ == '__main__':
    rospy.init_node('image_generator_node')
    cs_controller = AwesomeROSControllerIiwas()
    cs_controller.open_simulation()
    cs_controller.start_simulation()
    cs_controller.goto_joint('LEFT_ARM', np.array([0,-0.20,0,-1,0,1.1,0]),
                                np.array([3]))
    cs_controller.wait_for_goto('LEFT_ARM')
    cs_controller.grasp('LEFT_GRIPPER',15, 100)
    cs_controller.wait_for_grasp('LEFT_GRIPPER')

    while True:
        time.sleep(CAMERA_DELAY)
        cs_controller.step()
        cs_controller.publish()


