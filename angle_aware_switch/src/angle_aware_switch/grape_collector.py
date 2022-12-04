#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseArray, PoseStamped


class GrapeCollector:
    def __init__(self):
        # Number of Agents
        self._agentNum = rospy.get_param("agentNum")
        posestampedTopic = rospy.get_param(
            "~posestampedTopic", default="grape_posestamped"
        )
        output_topic = rospy.get_param("~output_topic", default="/all_grape")

        group_ns = rospy.get_param("~group_ns", default="bebop10")

        self._pub_all_pose = rospy.Publisher(output_topic, PoseArray, queue_size=1)
        for i in range(self._agentNum):
            topic_name = group_ns + str(i + 1) + "/" + posestampedTopic
            rospy.Subscriber(
                topic_name,
                PoseStamped,
                self.callback,
                queue_size=10,
            )
        self._pose_list = []

    #############################################################
    # callback
    #############################################################
    def callback(self, msg):
        self._pose_list.append(msg.pose)
        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.poses = self._pose_list
        self._pub_all_pose.publish(pose_array)


if __name__ == "__main__":
    rospy.init_node("GrapeCollector", anonymous=True)
    posecollector = GrapeCollector()
    rospy.spin()
