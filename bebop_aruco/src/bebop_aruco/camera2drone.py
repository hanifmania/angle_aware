#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")  ##ignore the warning due to the python2

from bebop_hatanaka_base import ros_utility

import rospy
import tf
import tf2_ros
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped, Pose

import numpy as np


class Camera2Drone:
    def __init__(self):
        input_topic = rospy.get_param("~input_topic", default="camera_control")
        input_posestamped_topic = rospy.get_param("~input_posestamped_topic", default="fromAR/camera_posestamped")
        output_topic = rospy.get_param("~output_topic", default="posestamped")

        self._camera2bebop_pose = None

        self._pub_posestamped = rospy.Publisher(output_topic, PoseStamped, queue_size=1)
        rospy.Subscriber(input_topic, Twist, self.camera_control_callback)
        rospy.Subscriber(
            input_posestamped_topic, PoseStamped, self.posestamped_callback
        )

    def posestamped_callback(self, world2camera_posestamped):
        if self._camera2bebop_pose is None:
            return
        world2camera_g = ros_utility.pose_to_g(world2camera_posestamped.pose)
        camera2bebop_g = ros_utility.pose_to_g(self._camera2bebop_pose)
        world2bebop_g = world2camera_g.dot(camera2bebop_g)
        pub_msg = ros_utility.g_to_posestamped(world2bebop_g)
        self._pub_posestamped.publish(pub_msg)

    def camera_control_callback(self, msg):
        camera_deg = msg.angular.y
        self._camera2bebop_pose = Pose()
        camera_orientation = tf.transformations.quaternion_from_euler(
            0, np.deg2rad(camera_deg), 0, "rxyz"
        )

        self._camera2bebop_pose.orientation.x = camera_orientation[0]
        self._camera2bebop_pose.orientation.y = camera_orientation[1]
        self._camera2bebop_pose.orientation.z = camera_orientation[2]
        self._camera2bebop_pose.orientation.w = camera_orientation[3]


if __name__ == "__main__":
    rospy.init_node("camera_2_drone", anonymous=True)
    node = Camera2Drone()
    rospy.spin()
