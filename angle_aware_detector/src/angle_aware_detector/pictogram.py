#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PoseStamped
from jsk_rviz_plugins.msg import Pictogram, PictogramArray


class ShowPictogram:
    def __init__(self):
        ################## param input
        input_detect_topic = rospy.get_param(
            "~input_detect_topic", default="object_detector/target_posestamped"
        )
        output_pictogram = rospy.get_param("~output_pictogram", default="grape")
        self._pictogram_param = rospy.get_param("/pictogram")
        self._world_tf = rospy.get_param("~world", default="world")

        self._pictogram_size = self._pictogram_param["size"]
        self._pictogram_character = self._pictogram_param["character"]
        self._pictogram_z_offset = self._pictogram_param["z_offset"]
        self._pictogram_rotate_speed = self._pictogram_param["rotate_speed"]
        ###################

        self._pictogram_array = PictogramArray()

        self._pub = rospy.Publisher(output_pictogram, PictogramArray, queue_size=1)
        rospy.Subscriber(input_detect_topic, PoseStamped, self.callback)

    #############################################################
    # callback
    #############################################################

    def callback(self, msg):
        pictogram_array = self.calc_pictogram_array(msg)
        self._pub.publish(pictogram_array)

    #############################################################
    # main function
    #############################################################
    def calc_pictogram_array(self, posestamped):

        pictogram = self.generate_pictogram(
            self._world_tf,
            posestamped.pose.position,
            self._pictogram_size,
            self._pictogram_character,
            self._pictogram_z_offset,
            self._pictogram_rotate_speed,
        )

        self._pictogram_array.header.frame_id = self._world_tf
        self._pictogram_array.header.stamp = rospy.Time.now()
        self._pictogram_array.pictograms.append(pictogram)
        return self._pictogram_array

    #############################################################
    # functions
    #############################################################
    def generate_pictogram(
        self, world_tf, position, size, character, z_offset, rotate_speed
    ):
        """ピクトグラムを生成

        Args:
            world_tf (str): _description_
            position (Position): _description_
            size (float): _description_
            character (str): _description_
            z_offset(float): 高さoffset. 他の描写と被らないようにするため
            rotate_speed(float): 回転速度

        Returns:
            Pictogram: _description_
        """
        color_rgba = [1, 0, 0, 1]

        msg = Pictogram()
        msg.action = Pictogram.ROTATE_X
        msg.header.frame_id = world_tf
        msg.header.stamp = rospy.Time.now()

        msg.pose.position = position
        msg.pose.position.z += z_offset
        msg.pose.orientation.w = 0.7
        msg.pose.orientation.x = 0
        msg.pose.orientation.y = -0.7
        msg.pose.orientation.z = 0
        msg.mode = Pictogram.PICTOGRAM_MODE
        msg.speed = rotate_speed
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
