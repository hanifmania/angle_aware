#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase

import rospy
from geometry_msgs.msg import PoseStamped, PoseArray

import numpy as np


class GrapeDetector:
    def __init__(self):
        output_topic = rospy.get_param("~output_topic", default="grape_posestamped")
        input_all_grape = rospy.get_param("~input_all_grape", default="/all_grape")
        self._world_tf = rospy.get_param("~world", default="world")
        grape_detector = rospy.get_param("/grape_detector")
        self._grape_positions = grape_detector["grape_positions"]
        self._clock = grape_detector["clock"]
        self._detect_radius = grape_detector["detect_radius"]
        self._grape_span = grape_detector["grape_span"]
        self._agent_base = AgentBase()
        self._all_grape = []
        self._pub = rospy.Publisher(output_topic, PoseStamped, queue_size=1)
        rospy.Subscriber(
            input_all_grape, PoseArray, self.all_grape_callback, queue_size=1
        )

    #############################################################
    # callback
    #############################################################

    def all_grape_callback(self, msg):
        positions = []
        orientations = []
        for pose in msg.poses:
            pos = [
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ]
            positions.append(pos)
            # q = [
            #     pose.orientation.x,
            #     pose.orientation.y,
            #     pose.orientation.z,
            #     pose.orientation.w,
            # ]
            # orientations.append(q)
        self._all_grape = np.array(positions)
        # self._all_orientation = np.array(orientations)

    #############################################################
    # functions
    #############################################################

    def main(self):
        position, _ = self._agent_base.get_my_pose()
        grape_position = self.detect_grape(position)
        if grape_position is None:
            return

        if self.observed_check(grape_position, self._all_grape, self._grape_span):
            return

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self._world_tf
        msg.pose.position.x = grape_position[0]
        msg.pose.position.y = grape_position[1]
        msg.pose.position.z = grape_position[2]  # [TODO] get in param
        self._pub.publish(msg)

    def detect_grape(self, position):
        """ぶどう検出器。dummy

        Args:
            position (list): drone position

        Returns:
            list: ぶどう位置xyz
        """
        if len(self._grape_positions) == 0:
            ### 未発見のぶどうがなければ何もしない
            return None

        detected_grape_id = None
        for i, grape_position in enumerate(self._grape_positions):
            if (grape_position[0] - position[0]) ** 2 + (
                grape_position[1] - position[1]
            ) ** 2 < self._detect_radius:
                ### ぶどうを発見
                detected_grape_id = i
                break

        if detected_grape_id is None:
            ### ぶどうが無ければ何もしない
            return None

        ### 見つけたぶどうの位置をpublishし、listから消す
        grape_position = self._grape_positions.pop(detected_grape_id)
        return grape_position

    def observed_check(self, grape_position, all_grape, grape_span):
        """既知のものかチェック

        Args:
            grape_position (list): new grape position xyz
            all_grape (ndarray): 既知のぶどうリスト
            grape_span (float): ぶどう間隔。この間隔以内のぶどうは同一とみなす

        Returns:
            bool: ぶどうが既知ならTrue
        """
        if len(all_grape) == 0:
            return False
        position = np.array(grape_position)
        dist = np.linalg.norm(position - all_grape, axis=1)
        min_dist = np.min(dist, axis=0)
        rospy.loginfo("observed check")
        rospy.loginfo(dist)
        return min_dist < grape_span

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
    node = GrapeDetector()
    node.spin()
