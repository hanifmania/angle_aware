#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from jsk_rviz_plugins.msg import Pictogram, PictogramArray

import numpy as np
import matplotlib.pyplot as plt


class ShowPictogram:
    def __init__(self):
        ################## param input
        input_detect_topic = rospy.get_param(
            "~input_detect_topic", default="object_detector/target_posestamped"
        )
        input_J_topic = rospy.get_param("~input_J_topic", default="angle_aware/J")
        output_pictogram = rospy.get_param("~output_pictogram", default="grape")
        self._pictogram_param = rospy.get_param("/pictogram")
        self._world_tf = rospy.get_param("~world", default="world")
        ###################

        self._posestamped_list = []
        self._J = 0
        self._list_id = -1
        self._J_0 = 1
        self._pictogram_array = PictogramArray()

        self._pub = rospy.Publisher(output_pictogram, PictogramArray, queue_size=1)
        rospy.Subscriber(input_detect_topic, PoseStamped, self.callback)
        rospy.Subscriber(input_J_topic, Float32, self.J_callback)

    #############################################################
    # callback
    #############################################################

    def callback(self, msg):
        self._posestamped_list.append(msg)

    def J_callback(self, msg):
        pictogram_array = self.calc_pictogram_array(msg.data)
        self._pub.publish(pictogram_array)

    #############################################################
    # main function
    #############################################################
    def calc_pictogram_array(self, J):
        if J > self._J:
            self.new_target(J)
        self._J = J

        target_posestamped = self._posestamped_list[self._list_id]
        pictogram = self.generate_pictogram(
            self._J,
            self._J_0,
            self._world_tf,
            target_posestamped.pose.position,
            self._pictogram_param["size"],
            self._pictogram_param["character"],
        )

        self._pictogram_array.header.frame_id = self._world_tf
        self._pictogram_array.header.stamp = rospy.Time.now()
        self._pictogram_array.pictograms[self._list_id] = pictogram
        return self._pictogram_array

    #############################################################
    # functions
    #############################################################
    def new_target(self, J):
        """Jが増加 = 新たなgrapeを探索している. 要素を一つ増やす

        Args:
            J (float): 重要度の和
        """
        self._J_0 = J
        self._list_id += 1
        self._pictogram_array.pictograms.append(Pictogram())

    def generate_pictogram(self, J, J_0, world_tf, position, size, character):
        """ピクトグラムを生成

        Args:
            J (float): 重要度
            J_0 (float): 重要度の初期値
            world_tf (str): _description_
            position (Position): _description_
            size (float): _description_
            character (str): _description_

        Returns:
            Pictogram: _description_
        """
        color_seed = J / J_0
        # color_rgba = plt.get_cmap("jet")(color_seed)
        color_rgba = [1, 0, 0, 1]

        msg = Pictogram()
        msg.action = Pictogram.ROTATE_X
        msg.header.frame_id = world_tf
        msg.header.stamp = rospy.Time.now()

        msg.pose.position = position
        msg.pose.orientation.w = 0.7
        msg.pose.orientation.x = 0
        msg.pose.orientation.y = -0.7
        msg.pose.orientation.z = 0
        msg.mode = Pictogram.PICTOGRAM_MODE
        msg.speed = 0.1
        msg.ttl = 60 * 10
        msg.size = size
        msg.color.r = color_rgba[0]
        msg.color.g = color_rgba[1]
        msg.color.b = color_rgba[2]
        msg.color.a = color_rgba[3]
        msg.character = character

        return msg


if __name__ == "__main__":
    rospy.init_node("template", anonymous=True)
    node = ShowPictogram()
    rospy.spin()
