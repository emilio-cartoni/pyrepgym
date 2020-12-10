import rospy
import numpy as np
import std_msgs.msg
import sensor_msgs.msg
from ias_coppelia_sim_iiwas.ros import ROSCSControllerIiwasAirHockey
from ias_coppelia_sim_iiwas.sensors import RealSense
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrepgym.envs.grid import make_grid
from pyrepgym.envs.PyRepEnv import macro_space
import tf2_ros
import geometry_msgs.msg

IMAGE_TOPIC_NAME = 'kai/has/to/look/up/the/final/topic/name'
CAMERA_DELAY = 5

class AwesomeROSControllerIiwas(ROSCSControllerIiwasAirHockey):

    def __init__(self, headless=True, verbose=True, time_step=0.05, auto_start=False, publishImage=False):
        self.publishImage = publishImage
        ROSCSControllerIiwasAirHockey.__init__(self, headless, verbose, time_step, auto_start)
        self.last_image = -10000
        self.cube_pos =[-1.0, 0.0, 0.191]
        self.objects = {}


    def _load_scene_components(self, **kwargs):
        ROSCSControllerIiwasAirHockey._load_scene_components(self, **kwargs)
        # load camera
        self.camera = RealSense.create(color=True, depth=False,
                                       position=[-0.9, 0., 1.1],
                                       orientation=[np.pi, 0., np.pi/2])

        self.camera.set_handle_explicitly()
        self.camera.set_model_collidable(False)
        # create object
        self.makeObject()
        handles, poses = make_grid(macro_space)


    def _start_ros_interface(self):
        ROSCSControllerIiwasAirHockey._start_ros_interface(self)

        if self.publishImage:
            # publisher for camera
            self.video_publisher = rospy.Publisher(IMAGE_TOPIC_NAME,
                                                   sensor_msgs.msg.Image,
                                                   queue_size=1)

        self.objpos_publisher = tf2_ros.TransformBroadcaster()


    def _publish(self):
        ROSCSControllerIiwasAirHockey._publish(self)

        
        if (self._t - self.last_image) > (CAMERA_DELAY - 0.001):
            # reset object if needed
            self.control_objects_limits()

            if self.publishImage:
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
            obj_pos = self.objects['cube'].get_pose()

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time(self._t)
            t.header.frame_id = "coppelia_origin"
            t.child_frame_id = "cube"
            t.transform.translation.x = obj_pos[0]
            t.transform.translation.y = obj_pos[1]
            t.transform.translation.z = obj_pos[2]
            t.transform.rotation.x = obj_pos[3]
            t.transform.rotation.y = obj_pos[4]
            t.transform.rotation.z = obj_pos[5]
            t.transform.rotation.w = obj_pos[6]

            self.objpos_publisher.sendTransform(t)


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
            if z < 0.18 or x < -1.1 or x > -0.9 or abs(y) > 0.40: #fallen off the table or too far
                self.objects[obj].set_pose(pose=self.cube_pos + [0, 0, 0, 1])



if __name__ == '__main__':
    rospy.init_node('coppelia_sim_iiwas_node')

    time_step = 0.05
    real_time = True

    cs_controller = AwesomeROSControllerIiwas(time_step=time_step)
    cs_controller.open_simulation()
    cs_controller.start_simulation()
    cs_controller.spin(real_time)
