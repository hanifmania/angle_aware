#!/usr/bin/env python
# -*- coding: utf-8 -*-

from angle_aware_control.central import Central

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
from scipy.stats import norm


class CentralwithCamera(Central):
    def __init__(self):
        super(CentralwithCamera, self).__init__()
        input_pitch_topic = rospy.get_param("~input_pitch_topic")
        rospy.Subscriber(input_pitch_topic, Float32MultiArray, self.pitch_callback)

    #############################################################
    # callback
    #############################################################
    def pitch_callback(self, msg):
        self._pitches = msg.data.reshape(-1)

    #############################################################
    # functions
    #############################################################

    def performance_function(self, pos, pitch, yaw, psi_grid):
        dist_map = np.sqrt(
            (pos[0] - psi_grid[0]) ** 2
            + (pos[1] - psi_grid[1]) ** 2
            + (pitch - psi_grid[2]) ** 2
            + (yaw - psi_grid[3]) ** 2
        )
        return norm.pdf(dist_map, scale=self._sigma) * np.sqrt(2 * np.pi) * self._sigma

    def update_psi(self):
        all_positions = self._agent_base.get_all_positions()
        yaws = self._agent_base.get_all_yaw()

        performance_functions = [
            self.performance_function(pos, self._psi_grid, pitch, yaw)
            for pos, pitch, yaw in zip(all_positions, self._pitches, yaws)
        ]
        dist2 = np.stack(performance_functions)
        h_max = dist2.max(axis=0)
        self._psi -= self._delta_decrease * h_max * self._psi * self._dt
        # print(np.sum(self._delta_decrease * h_max * self._psi * self._dt))
        self._psi = (0 < self._psi) * self._psi  ## the minimum value is 0


if __name__ == "__main__":
    rospy.init_node("central", anonymous=True)
    node = CentralwithCamera()
    node.spin()
