#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase

import rospy
from geometry_msgs.msg import PoseStamped


class GrapeDetector:
    def __init__(self):
        output_topic = rospy.get_param("~output_topic", default="grape_posestamped")
        self._world_tf = rospy.get_param("~world", default="world")
        grape_detector = rospy.get_param("/grape_detector")
        self._grape_positions = grape_detector["grape_positions"]
        self._clock = grape_detector["clock"]
        self._detect_radius = grape_detector["detect_radius"]
        self._agent_base = AgentBase()

        self._pub = rospy.Publisher(output_topic, PoseStamped, queue_size=1)

    #############################################################
    # callback
    #############################################################

    def callback(self, msg):
        self._pub.publish(msg)

    #############################################################
    # functions
    #############################################################

    def main(self):
        if len(self._grape_positions) == 0:
            ### 未発見のぶどうがなければ何もしない
            return
        position, _ = self._agent_base.get_my_pose()

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
            return

        ### 見つけたぶどうの位置をpublishし、listから消す
        grape_position = self._grape_positions.pop(detected_grape_id)

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self._world_tf
        msg.pose.position.x = grape_position[0]
        msg.pose.position.y = grape_position[1]
        msg.pose.position.z = 0  # [TODO] get in param
        self._pub.publish(msg)

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
