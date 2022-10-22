#!/usr/bin/env python
# -*- coding: utf-8 -*-
from angle_aware_control.field_generator import FieldGenerator

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray

import numpy as np
import matplotlib.pyplot as plt


class ShowPointCloud2d:
    def __init__(self):
        field_param_name = rospy.get_param("~field_param_name")
        field_param = rospy.get_param(field_param_name)
        input_field_topic = rospy.get_param("~input_field_topic")
        output_point_cloud_topic = rospy.get_param(
            "~output_point_cloud_topic", "psi_pointcloud"
        )
        self.WORLD_TF = rospy.get_param("~world", default="world")

        ### point cloud 作成共通部分
        field_generator = FieldGenerator(field_param)
        field_grid = field_generator.generate_grid(sparse=False)
        self._x_row = field_grid[0].reshape([-1, 1])
        self._y_row = field_grid[1].reshape([-1, 1])
        msg = PointCloud2()
        msg.header.frame_id = self.WORLD_TF
        msg.height = 1
        msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("r", 12, PointField.FLOAT32, 1),
            PointField("g", 16, PointField.FLOAT32, 1),
            PointField("b", 20, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.is_dense = True
        self._point_cloud_msg_base = msg
        self._phi_max = None

        self._pub_pointcloud = rospy.Publisher(
            output_point_cloud_topic, PointCloud2, queue_size=1
        )
        rospy.Subscriber(input_field_topic, Float32MultiArray, self.callback)

    #############################################################
    # callback
    #############################################################
    def callback(self, msg):
        self.publish_pointcloud(msg)

    #############################################################
    # functions
    #############################################################
    def publish_pointcloud(self, float32):
        """Float32Multiarrayから point cloudを生成。値を色、z方向の両方で表現

        Args:
            float32 (Float32MultiArray): 2d map
        Note:
         points = [
           [x1, y1, phi1, r1, g1, b1],
           [x2, y2, phi2, r2, g2, b2],
                   :
           [xn, yn, phin, rn, gn, bn]
         ]
        """
        phi = np.array(float32.data).reshape([-1, 1])
        if self._phi_max is None:
            self._phi_max = np.max(phi)
        normalized_phi = phi / self._phi_max
        rgba_phi = plt.get_cmap("jet")(normalized_phi).squeeze()
        z = normalized_phi
        points = np.hstack([self._x_row, self._y_row, z, rgba_phi[:, 0:3]])

        points = points.astype("float32")
        n = len(points)
        self._point_cloud_msg_base.width = n
        self._point_cloud_msg_base.row_step = self._point_cloud_msg_base.point_step * n
        self._point_cloud_msg_base.data = points.tobytes()
        self._pub_pointcloud.publish(self._point_cloud_msg_base)


if __name__ == "__main__":
    rospy.init_node("PointCloud2d", anonymous=True)
    node = ShowPointCloud2d()
    rospy.spin()
