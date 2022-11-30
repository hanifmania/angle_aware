#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase
from coverage_util.numpy2multiarray import multiarray2numpy

import rospy
from std_msgs.msg import Float32MultiArray

import numpy as np


class SavePsi:
    def __init__(self):
        self._file_path = rospy.get_param("~output_psi_path")
        input_topic = rospy.get_param("~input_topic", default="psi")

        self._agent_base = AgentBase()
        rospy.Subscriber(input_topic, Float32MultiArray, self.callback)
        rospy.wait_for_message(input_topic, Float32MultiArray)
        rospy.sleep(1)
        np.save(self._file_path, self._psi)
        pos = self._agent_base.get_all_positions()
        print("x: ", pos[0][0])
        print("y: ", pos[0][1])
        print("z: ", pos[0][2])

    #############################################################
    # callback
    #############################################################

    def callback(self, msg):
        self._psi = multiarray2numpy(float, np.float32, msg)

    #############################################################
    # functions
    #############################################################


if __name__ == "__main__":
    rospy.init_node("SavePsi", anonymous=True)
    node = SavePsi()
