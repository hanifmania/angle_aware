#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bebop_hatanaka_base.agent_base import AgentBase
from coverage_util.field_generator import FieldGenerator
from coverage_util.numpy2multiarray import numpy2multiarray
from angle_aware_control.psi_generator_no_jax import zeta_func

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray, Bool, Float32
import numpy as np
from scipy.stats import norm


class Central:
    def __init__(self):
        self._clock = rospy.get_param("central_clock")
        angle_aware_params = rospy.get_param("angle_aware")
        self._sigma = angle_aware_params["sigma"]
        self._delta_decrease = angle_aware_params["delta_decrease"]
        phi_param = angle_aware_params["rviz_phi"]
        ref_z = rospy.get_param("agents/ref_z")
        output_phi_topic = rospy.get_param("~output_phi_topic", default="phi")
        # output_J_topic = rospy.get_param("~output_J_topic", default="J")

        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path("angle_aware_control")
        # data_dir = pkg_path + "/data/input/"
        # data_dir = rospy.get_param("~npy_data_dir")
        # self._phi_path = data_dir + phi_param["npy_name"]

        self._dt = 1.0 / self._clock
        self._t  = None
        phi_generator = FieldGenerator(phi_param)
        self._phi_grid = phi_generator.generate_grid()
        self._phi_generator = phi_generator
        self._phi_generator = phi_generator
        self._zeta = zeta_func(self._phi_grid, ref_z)

        # phi_param = angle_aware_params["phi"]
        # phi_generator = FieldGenerator(phi_param)
        # phi_shape = phi_generator.get_shape()
        # rospy.loginfo(phi_shape)
        self.load_phi()


        self._pub_phi = rospy.Publisher(
            output_phi_topic, Float32MultiArray, queue_size=1
        )

        rospy.Subscriber("takeoffstatus", Bool, self.take_off_callback)
        self._agent_base = AgentBase()

    #############################################################
    # functions
    #############################################################
    def publish_phi(self):
        phi_multiarray = numpy2multiarray(Float32MultiArray, self._phi)
        self._pub_phi.publish(phi_multiarray)

    def performance_function(self, pos, phi_grid):
        dist_map = np.sqrt((pos[0] - phi_grid[0]) ** 2 + (pos[1] - phi_grid[1]) ** 2)
        return norm.pdf(dist_map, scale=self._sigma) * np.sqrt(2 * np.pi) * self._sigma

    def update_phi(self):
        all_positions = self._agent_base.get_all_positions()
        performance_functions = [
            self.performance_function(pos, self._zeta) for pos in all_positions
        ]
        dist2 = np.stack(performance_functions)
        h_max = dist2.max(axis=0)

        t = rospy.Time().now()
        if self._t is None:
            self._t = t
        dt = t - self._t 
        self._t = t
        self._phi -= self._delta_decrease * h_max * self._phi * dt.to_sec()
        # print(np.sum(self._delta_decrease * h_max * self._phi * self._dt))
        self._phi = (0 < self._phi) * self._phi  ## the minimum value is 0

    def take_off_callback(self, msg):
        ### phiのreset
        self.load_phi()

    def load_phi(self):
        self._phi = self._phi_generator.generate_phi()
        
    #############################################################
    # spin
    #############################################################
    def spin(self):
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            self.publish_phi()
            if self._agent_base.is_main_ok():
                self.update_phi()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("central", anonymous=True)
    node = Central()
    node.spin()
