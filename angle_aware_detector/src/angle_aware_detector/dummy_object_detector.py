#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase

import rospy
from geometry_msgs.msg import PoseStamped, PoseArray

import numpy as np


class DummyObjectDetector:
    def __init__(self):
        output_topic = rospy.get_param(
            "~output_topic", default="object_detector/posestamped"
        )
        self._world_tf = rospy.get_param("~world", default="world")
        grape_detector = rospy.get_param("/dummy_object_detector")
        self._object_positions = grape_detector["positions"]
        self._clock = grape_detector["clock"]
        self._detect_radius = grape_detector["detect_radius"]
        self._agent_base = AgentBase()
        self._pub = rospy.Publisher(output_topic, PoseStamped, queue_size=1)

    #############################################################
    # publish
    #############################################################
    def publish_posestamped(self, position):
        """物体位置をpublish

        Args:
            position (ndarray): [x, y, z]
        """
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self._world_tf
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]  # [TODO] get in param
        msg.pose.orientation.w = 1
        self._pub.publish(msg)

    #############################################################
    # functions
    #############################################################
    def main(self):
        position, _ = self._agent_base.get_my_pose()
        grape_position = self.detect_grape(position)
        if grape_position is None:
            return
        self.publish_posestamped(grape_position)

    #############################################################
    # functions
    #############################################################
    def detect_grape(self, position):
        """ぶどう検出器。dummy

        Args:
            position (list): drone position

        Returns:
            list: ぶどう位置xyz
        """
        detected_grape_id = None
        for i, grape_position in enumerate(self._object_positions):
            dist = np.linalg.norm(
                [grape_position[0] - position[0], grape_position[1] - position[1]]
            )
            if dist < self._detect_radius:
                ### ぶどうを発見
                detected_grape_id = i
                break

        if detected_grape_id is None:
            ### ぶどうが無ければ何もしない
            return None

        ### 見つけたぶどうの位置をpublish
        grape_position = self._object_positions[detected_grape_id]
        return grape_position

    #############################################################
    # spin
    #############################################################

    def spin(self):
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            if self._agent_base.is_main_ok():
                self.main()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("grape_detector", anonymous=True)
    node = DummyObjectDetector()
    node.spin()
