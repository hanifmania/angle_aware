#!/usr/bin/env python
# -*- coding: utf-8 -*-
from angle_aware_control.agent_with_unom import Agent

from angle_aware_control.myqp import MyQP

# from angle_aware_avoid_tree.myqp import QPAvoidTree

import rospy
from std_msgs.msg import Bool

import numpy as np


class PatrolAgent:
    def __init__(self):
        input_flag_topic = rospy.get_param(
            "~input_flag_topic", default="angle_aware_mode"
        )
        trees = rospy.get_param("/trees")

        self._agent = Agent(MyQP)
        self._agent._qp.set_obstacle_avoidance_param(trees)

        self._is_patrol = True
        rospy.Subscriber(input_flag_topic, Bool, self.callback)

    ###################################################################
    ### callback
    ###################################################################
    def callback(self, msg):
        self._is_patrol = not msg.data

    ###################################################################
    ### spin
    ###################################################################
    def spin(self):
        rate = rospy.Rate(self._agent._clock)
        while not rospy.is_shutdown():
            if self._agent._agent_base.is_main_ok() and self._is_patrol:
                self._agent.main_control()

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("patrol_agent", anonymous=True)
    node = PatrolAgent()
    node.spin()
