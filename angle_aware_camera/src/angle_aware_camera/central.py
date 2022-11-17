#!/usr/bin/env python
# -*- coding: utf-8 -*-


from bebop_hatanaka_base.agent_base import AgentBase
from coverage_util.field_generator import FieldGenerator
from coverage_util.numpy2multiarray import numpy2multiarray
from angle_aware_camera.jax_func import (
    zeta4d,
    calc_J,
    calc_phi,
)

import rospy
from std_msgs.msg import Float32MultiArray, Bool, Float32

from jax import device_put
import jax.numpy as np


class Central:
    def __init__(self):
        self._clock = rospy.get_param("central_clock")
        angle_aware_params = rospy.get_param("angle_aware")
        self._sigma = angle_aware_params["sigma"]
        self._delta_decrease = angle_aware_params["delta_decrease"]
        phi_param = angle_aware_params["phi"]
        output_phi_topic = rospy.get_param("~output_phi_topic", default="phi")
        output_J_topic = rospy.get_param("~output_J_topic", default="J")

        ref_z = rospy.get_param("agents/ref_z")
        phi_param = angle_aware_params["phi"]
        self._phi_generator = FieldGenerator(phi_param)

        phi_grid = self._phi_generator.generate_grid()
        rospy.loginfo(phi_grid)
        zeta = zeta4d(phi_grid, ref_z)

        self._dt = 1.0 / self._clock
        self._pub_phi = rospy.Publisher(
            output_phi_topic, Float32MultiArray, queue_size=1
        )
        self._pub_J = rospy.Publisher(output_J_topic, Float32, queue_size=1)

        self.load_phi()
        rospy.Subscriber("takeoffstatus", Bool, self.take_off_callback)
        self._agent_base = AgentBase()

        self._zeta = device_put(zeta)
        self._sigma = device_put(self._sigma)
        self._delta_decrease = device_put(self._delta_decrease)
        self._dt = device_put(self._dt)

    #############################################################
    # callback
    #############################################################

    def take_off_callback(self, msg):
        ### phiのreset
        self.load_phi()

    #############################################################
    # functions
    #############################################################
    def publish_phi(self):
        phi_multiarray = numpy2multiarray(Float32MultiArray, self._phi)
        self._pub_phi.publish(phi_multiarray)

    def publish_J(self):
        J = calc_J(self._phi) * self._phi_generator.get_point_dense()
        self._pub_J.publish(J)

    def update_phi(self):
        all_positions = self._agent_base.get_all_positions()
        yaws = self._agent_base.get_all_yaw()
        cameras = self._agent_base.get_all_camera()

        all_positions = device_put(all_positions)
        cameras = device_put(cameras)
        yaws = device_put(yaws)

        self._phi = calc_phi(
            all_positions,
            np.deg2rad(cameras),
            yaws,
            self._zeta,
            self._sigma,
            self._phi,
            self._delta_decrease,
            self._dt,
        ).block_until_ready()

    def load_phi(self):
        self._phi = self._phi_generator.generate_phi()
        self._phi = device_put(self._phi)

    #############################################################
    # spin
    #############################################################
    def spin(self):
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            self.publish_phi()
            self.publish_J()
            if self._agent_base.is_main_ok():
                self.update_phi()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("central", anonymous=True)
    node = Central()
    node.spin()
