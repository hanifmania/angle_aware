#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase
from angle_aware_control.myqp import MyQP
from coverage_util.field_generator import FieldGenerator
from coverage_util.numpy2multiarray import multiarray2numpy

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray


class Agent:
    def __init__(self):
        self.agentID = rospy.get_param("agentID", default=-1)
        self._camera_deg = rospy.get_param("initial_pose/camera_deg")
        field_cbf = rospy.get_param("/field_cbf")
        agents_param = rospy.get_param("/agents")
        angle_aware_params = rospy.get_param("/angle_aware")
        input_psi_topic = rospy.get_param("~input_psi_topic", default="/psi")
        self._clock = agents_param["agent_manager_clock"]
        self._umax = agents_param["u_max"]
        collision_distance = agents_param["collision_distance"]
        self._kp_z = agents_param["kp_z"]
        self._ref_z = agents_param["ref_z"]
        self._kp_yaw = agents_param["kp_yaw"]
        self._ref_yaw = agents_param["ref_yaw"]
        psi_param = angle_aware_params["psi"]

        psi_generator = FieldGenerator(psi_param)
        self._psi = None  # psi_generator.generate_phi()
        self._psi_grid = psi_generator.generate_grid()

        self._agent_base = AgentBase(self.agentID)
        self._qp = MyQP(field_cbf, collision_distance, angle_aware_params)

        rospy.Subscriber(input_psi_topic, Float32MultiArray, self.psi_callback)
        self._agent_base.wait_pose_stamped()
        self._agent_base.publish_camera_control(self._camera_deg)

    #############################################################
    # callback
    #############################################################
    def psi_callback(self, msg):
        self._psi = multiarray2numpy(float, np.float32, msg)

    ###################################################################
    ### main
    ###################################################################
    def main_control(self):
        my_position, my_orientation = self._agent_base.get_my_pose()
        neighbor_positions = self._agent_base.get_neighbor_positions()
        yaw = self._agent_base.get_my_yaw()
        #### joy input
        # uh_x, uh_y, uh_z, uh_w, uh_camera = self._agent_base.get_uh()

        ##########################################
        #  generate ux,uy,uz. You can write any code here
        ##########################################
        ## u_nom = [0,0]
        world_ux = 0  # uh_x
        world_uy = 0  # uh_y

        ## 高度を一定に保つ
        world_uz = self._kp_z * (self._ref_z - my_position[2])

        ## 姿勢を一定に保つ
        diff_rad = self._ref_yaw - yaw
        if diff_rad > np.pi:
            diff_rad -= 2 * np.pi
        elif diff_rad < -np.pi:
            diff_rad += 2 * np.pi
        omega_z = self._kp_yaw * diff_rad
        # omega_z = self._ref_yaw
        ##########################################

        ### x y z field limitation and collision avoidance with CBF
        u_nom = np.array([world_ux, world_uy])
        u_opt, w = self._qp.solve(
            u_nom, my_position[:2], neighbor_positions[:, :2], self._psi_grid, self._psi
        )
        world_ux, world_uy = u_opt
        world_ux, world_uy = self.velocity_limitation(world_ux, world_uy)
        self._agent_base.publish_command_from_world_vel(
            world_ux, world_uy, world_uz, omega_z
        )
        # rospy.loginfo(
        #     "agent {}, u: {:.3f}, {:.3f}".format(self.agentID, world_ux, world_uy)
        # )

    ###################################################################
    ### spin
    ###################################################################
    def spin(self):
        # self._clock=1
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            if self._agent_base.is_main_ok() and self._psi is not None:
                self.main_control()
            rate.sleep()

    ###################################################################
    ### functions
    ###################################################################
    def velocity_limitation(self, world_ux, world_uy):
        world_ux = np.clip(world_ux, -self._umax, self._umax)
        world_uy = np.clip(world_uy, -self._umax, self._umax)
        return world_ux, world_uy


if __name__ == "__main__":
    rospy.init_node("agent", anonymous=True)
    agent = Agent()
    agent.spin()
