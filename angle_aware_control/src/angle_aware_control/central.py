#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bebop_hatanaka_base.agent_base import AgentBase
from angle_aware_control.field_generator import FieldGenerator
from angle_aware_control.numpy2multiarray import numpy2multiarray
import rospy
import rospkg
from std_msgs.msg import Float32MultiArray, Bool, Float32
import numpy as np
from scipy.stats import norm


class Central:
    def __init__(self):
        self._clock = rospy.get_param("central_clock")
        ns = "angle_aware/"
        angle_aware_params = rospy.get_param("angle_aware")
        self._sigma = angle_aware_params["sigma"]
        self._delta_decrease = angle_aware_params["delta_decrease"]
        psi_param = angle_aware_params["psi"]
        output_psi_topic = rospy.get_param("~output_psi_topic", default="psi")
        output_J_topic = rospy.get_param("~output_J_topic", default="J")
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("angle_aware_control")
        data_dir = pkg_path + "/data/input/"
        self._file_path = data_dir + "psi.npy"

        self._dt = 1.0 / self._clock
        psi_generator = FieldGenerator(psi_param)
        self._psi_grid = psi_generator.generate_grid()
        self._psi_generator = psi_generator
        self._pub_psi = rospy.Publisher(
            output_psi_topic, Float32MultiArray, queue_size=1
        )
        self._pub_J = rospy.Publisher(output_J_topic, Float32, queue_size=1)

        self.load_psi()
        rospy.Subscriber("takeoffstatus", Bool, self.take_off_callback)
        self._agent_base = AgentBase()

    #############################################################
    # functions
    #############################################################
    def publish_psi(self):
        psi_multiarray = numpy2multiarray(Float32MultiArray, self._psi)
        self._pub_psi.publish(psi_multiarray)

    def publish_J(self):
        J = np.sum(self._psi)
        self._pub_J.publish(J)

    def performance_function(self, pos, psi_grid):
        dist_map = np.sqrt((pos[0] - psi_grid[0]) ** 2 + (pos[1] - psi_grid[1]) ** 2)
        return norm.pdf(dist_map, scale=self._sigma) * np.sqrt(2 * np.pi) * self._sigma

    def update_psi(self):
        all_positions = self._agent_base.get_all_positions()
        performance_functions = [
            self.performance_function(pos, self._psi_grid) for pos in all_positions
        ]
        dist2 = np.stack(performance_functions)
        h_max = dist2.max(axis=0)
        self._psi -= self._delta_decrease * h_max * self._psi * self._dt
        # print(np.sum(self._delta_decrease * h_max * self._psi * self._dt))
        self._psi = (0 < self._psi) * self._psi  ## the minimum value is 0

    def take_off_callback(self, msg):
        ### psiのreset
        self.load_psi()

    def load_psi(self):
        self._psi = np.load(self._file_path).reshape(self._psi_generator.get_shape())

    #############################################################
    # spin
    #############################################################
    def spin(self):
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            if self._agent_base.is_main_ok():
                self.update_psi()
                self.publish_psi()
                self.publish_J()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("central", anonymous=True)
    node = Central()
    node.spin()
