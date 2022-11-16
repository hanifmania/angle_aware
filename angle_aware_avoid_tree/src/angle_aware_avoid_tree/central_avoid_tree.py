#!/usr/bin/env python
# -*- coding: utf-8 -*-

from angle_aware_control.central import Central

import rospy
from std_msgs.msg import Float32MultiArray, Bool, Float32
import numpy as np
from scipy.stats import norm


class CentralAvoidTree(Central):
    def __init__(self):
        super(CentralAvoidTree, self).__init__()
        self._trees = rospy.get_param("trees")

    #############################################################
    # functions
    #############################################################
    def load_psi(self):
        super(CentralAvoidTree, self).load_psi()
        self._psi = np.load(self._psi_path).reshape(self._psi_generator.get_shape())


if __name__ == "__main__":
    rospy.init_node("central", anonymous=True)
    node = CentralAvoidTree()
    node.spin()
