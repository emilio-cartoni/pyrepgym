import rospy
import numpy as np
import std_msgs.msg
import sensor_msgs.msg
from ias_coppelia_sim_iiwas.ros import ROSCSControllerIiwasAirHockey
from ias_coppelia_sim_iiwas.sensors import RealSense
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

IMAGE_TOPIC_NAME = 'kai/has/to/look/up/the/final/topic/name'
OBJPOS_TOPIC_NAME = 'kai/has/to/look/up/another/final/topic/name'
CAMERA_DELAY = 5

class AwesomeROSControllerIiwas(ROSCSControllerIiwasAirHockey):

    def __init__(self, headless=False, verbose=True, time_step=0.05, auto_start=False):
        ROSCSControllerIiwasAirHockey.__init__(self, headless, verbose, time_step, auto_start)
        self.last_image = -10000
        self.cube_pos =[-1.0, -0.15, 0.25]
        self.objects = {}

    def _load_scene_components(self, **kwargs):
        ROSCSControllerIiwasAirHockey._load_scene_components(self, **kwargs)
        # load camera
        self.camera = RealSense.create(color=True, depth=False,
                                       position=[-0.9, 0.1, 1.1],
                                       orientation=[np.pi, 0., 0.])

        self.camera.set_handle_explicitly()
        # create object
        #self.makeObject()


    def _start_ros_interface(self):
        ROSCSControllerIiwasAirHockey._start_ros_interface(self)
        # publisher for camera
        self.video_publisher = rospy.Publisher(IMAGE_TOPIC_NAME,
                                               sensor_msgs.msg.Image,
                                               queue_size=1)
        # publisher for object position
        self.objpos_publisher = rospy.Publisher(OBJPOS_TOPIC_NAME,
                                               sensor_msgs.msg.Image, #to be changed
                                               queue_size=1)


    def _publish(self):
        ROSCSControllerIiwasAirHockey._publish(self)

        
        if (self._t - self.last_image) > (CAMERA_DELAY - 0.001):
            # reset object if needed
            #self.control_objects_limits()
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

            # publish object position
            # TBD

        else:
            print(self._t)

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

    def control_objects_limits(self):
        '''
        reset positions if an object goes out of the limits
        '''
        for obj in self.objects:
            x, y, z = self.objects[obj].get_position()
            if z < 0.20 or x > -0.9 or abs(y) > 0.44: #fallen off the table or too far
                self.objects[obj].set_pose(pose=self.cube_pos + [0, 0, 0, 1])



if __name__ == '__main__':
    rospy.init_node('coppelia_sim_iiwas_node')

    time_step = 0.05
    real_time = True

    cs_controller = AwesomeROSControllerIiwas(time_step=time_step)
    cs_controller.open_simulation()
    cs_controller.spin(real_time)
