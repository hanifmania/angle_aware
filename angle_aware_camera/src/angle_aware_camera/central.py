#!/usr/bin/env python
# -*- coding: utf-8 -*-


from bebop_hatanaka_base.agent_base import AgentBase
from coverage_util.field_generator import FieldGenerator
from coverage_util.numpy2multiarray import numpy2multiarray
from angle_aware_camera.jax_func import zeta4d, nearest_dist, performance_function

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
        phi_param = angle_aware_params["phi"]
        output_phi_topic = rospy.get_param("~output_phi_topic", default="phi")
        output_J_topic = rospy.get_param("~output_J_topic", default="J")

        ref_z = rospy.get_param("agents/ref_z")
        phi_param = angle_aware_params["phi"]
        self._phi_generator = FieldGenerator(phi_param)

        phi_grid = self._phi_generator.generate_grid()
        self._zeta_grid = zeta4d(phi_grid, ref_z)

        self._dt = 1.0 / self._clock
        self._pub_phi = rospy.Publisher(
            output_phi_topic, Float32MultiArray, queue_size=1
        )
        self._pub_J = rospy.Publisher(output_J_topic, Float32, queue_size=1)

        self.load_phi()
        rospy.Subscriber("takeoffstatus", Bool, self.take_off_callback)
        self._agent_base = AgentBase()

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
    def publish_phi(self):
        phi_multiarray = numpy2multiarray(Float32MultiArray, self._phi)
        self._pub_phi.publish(phi_multiarray)

    def publish_J(self):
        J = np.sum(self._phi)
        self._pub_J.publish(J)

    def update_phi(self):
        all_positions = self._agent_base.get_all_positions()
        yaws = self._agent_base.get_all_yaw()

        performance_functions = [
            performance_function(self._sigma, pos, self._zeta_grid, pitch, yaw)
            for pos, pitch, yaw in zip(all_positions, self._pitches, yaws)
        ]
        dist2 = np.stack(performance_functions)
        h_max = dist2.max(axis=0)
        self._phi -= self._delta_decrease * h_max * self._phi * self._dt
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
            self.publish_J()
            if self._agent_base.is_main_ok():
                self.update_phi()
            rate.sleep()

    def __init__(self):
        super(CentralwithCamera, self).__init__()

        self._phi_grid = zeta4d(self._phi_grid)
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

    def performance_function(self, pos, pitch, yaw, phi_grid):
        dist_map = np.sqrt(
            (pos[0] - phi_grid[0]) ** 2
            + (pos[1] - phi_grid[1]) ** 2
            + (pitch - phi_grid[2]) ** 2
            + (yaw - phi_grid[3]) ** 2
        )
        return norm.pdf(dist_map, scale=self._sigma) * np.sqrt(2 * np.pi) * self._sigma

    def update_phi(self):
        all_positions = self._agent_base.get_all_positions()
        yaws = self._agent_base.get_all_yaw()

        performance_functions = [
            self.performance_function(pos, self._phi_grid, pitch, yaw)
            for pos, pitch, yaw in zip(all_positions, self._pitches, yaws)
        ]
        dist2 = np.stack(performance_functions)
        h_max = dist2.max(axis=0)
        self._phi -= self._delta_decrease * h_max * self._phi * self._dt
        # print(np.sum(self._delta_decrease * h_max * self._phi * self._dt))
        self._phi = (0 < self._phi) * self._phi  ## the minimum value is 0


if __name__ == "__main__":
    rospy.init_node("central", anonymous=True)
    node = CentralwithCamera()
    node.spin()
