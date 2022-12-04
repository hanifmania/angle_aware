#!/usr/bin/env python
# -*- coding: utf-8 -*-
from angle_aware_control.agent_with_unom import AgentWithUnom
from angle_aware_control.myqp import MyQP

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

import numpy as np


class AngleAwareAgent:
    def __init__(self):
        flag_topic = rospy.get_param("~flag_topic", default="angle_aware_mode")
        input_detect_topic = rospy.get_param(
            "~input_detect_topic", default="grape_posestamped"
        )
        self._target_field_radius = rospy.get_param(
            "/grape_detector/target_field_radius", default=None
        )
        self._threshold = rospy.get_param("/grape_detector/threshold", default=None)
        self._agent = AgentWithUnom(MyQP)

        self._is_angle_aware = True
        self._target_field = 0
        self._grape_queue = []
        self._pub_flag = rospy.Publisher(flag_topic, Bool, queue_size=1)
        rospy.Subscriber(input_detect_topic, PoseStamped, self.grape_callback)

    ###################################################################
    ### callback
    ###################################################################
    def grape_callback(self, msg):
        self._grape_queue.append(msg)

    ###################################################################
    ### function
    ###################################################################

    def extract_target_field(self, posestamped, r, psi_grid):
        """ぶどうを中心に半径rの領域を被覆領域とする

        Args:
            posestamped (PoseStamped): ぶどう位置
            r (float): _description_
            psi_grid (ndarray): _description_

        Returns:
            ndarray: bool
        """
        x = posestamped.pose.position.x
        y = posestamped.pose.position.y
        return ((x - psi_grid[0]) ** 2 + (y - psi_grid[1]) ** 2) < r

    def judge_angle_aware(self):
        """angle awareかpatrolかを判断. また target filedの生成も行う

        Returns:
            bool: true if angle_aware
        """

        self._target_psi = self._target_field * self._agent._psi
        angle_aware_mode = np.sum(self._target_psi) > self._threshold
        if angle_aware_mode:
            ### まだangle awareすべき
            return True
        if len(self._grape_queue) == 0:
            ### もう見るべきぶどうが無い
            self._target_field = 0
            return False

        ### target fieldを新しくして再検証
        grape_posestamped = self._grape_queue.pop()
        self._target_field = self.extract_target_field(
            grape_posestamped, self._target_field_radius, self._agent._psi_grid
        )
        return self.judge_angle_aware()

    ###################################################################
    ### spin
    ###################################################################
    def spin(self):
        rate = rospy.Rate(self._agent._clock)
        while not rospy.is_shutdown():
            if self._agent._agent_base.is_main_ok():
                is_angle_aware = self.judge_angle_aware()
                self._pub_flag.publish(is_angle_aware)
                if is_angle_aware:
                    self._agent.main_control(self._target_psi)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("patrol_agent", anonymous=True)
    node = AngleAwareAgent()
    node.spin()
