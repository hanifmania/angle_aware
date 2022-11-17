#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase
from angle_aware_control.myqp import MyQP
from coverage_util.field_generator import FieldGenerator
from coverage_util.numpy2multiarray import multiarray2numpy
from angle_aware_camera.jax_func import (
    zeta4d,
    calc_J,
    calc_phi,
)


import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray


class Agent:
    def __init__(self, myqp):
        self.agentID = rospy.get_param("agentID", default=-1)
        self._camera_deg = rospy.get_param("initial_pose/camera_deg")
        field_cbf = rospy.get_param("/field_cbf")
        agents_param = rospy.get_param("/agents")
        angle_aware_params = rospy.get_param("/angle_aware")
        input_phi_topic = rospy.get_param("~input_phi_topic", default="/phi")
        self._clock = agents_param["agent_manager_clock"]
        self._umax = agents_param["u_max"]
        collision_distance = rospy.get_param("collision_distance")
        self._kp_z = agents_param["kp_z"]
        self._ref_z = agents_param["ref_z"]
        phi_param = angle_aware_params["phi"]

        phi_generator = FieldGenerator(phi_param)
        self._phi = None  # phi_generator.generate_phi()
        phi_grid = phi_generator.generate_grid()
        self._zeta_grid = zeta4d(phi_grid, self._ref_z)

        self._agent_base = AgentBase(self.agentID)
        self._qp = myqp(field_cbf, collision_distance, angle_aware_params)

        rospy.Subscriber(input_phi_topic, Float32MultiArray, self.phi_callback)
        self._agent_base.publish_camera_control(self._camera_deg)

    #############################################################
    # callback
    #############################################################
    def phi_callback(self, msg):
        self._phi = multiarray2numpy(float, np.float32, msg)

    ###################################################################
    ### main
    ###################################################################
    def main_control(self):
        my_position, my_orientation = self._agent_base.get_my_pose()
        neighbor_positions = self._agent_base.get_neighbor_positions()
        yaws = self._agent_base.get_all_yaw()
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

        ##########################################

        ### x y z field limitation and collision avoidance with CBF
        u_nom = np.array([world_ux, world_uy, 0, 0])
        u_opt, w = self._qp.solve(
            u_nom,
            my_position[:2],
            neighbor_positions[:, :2],
            self._zeta_grid,
            self._phi,
            yaws,
        )
        world_ux, world_uy, omega_z, camera_vel = u_opt
        world_ux, world_uy = self.velocity_limitation(world_ux, world_uy)
        self._agent_base.publish_command_from_world_vel(
            world_ux, world_uy, world_uz, omega_z
        )

    ###################################################################
    ### spin
    ###################################################################
    def spin(self):
        # self._clock=1
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            if self._agent_base.is_main_ok() and self._phi is not None:
                self.main_control()
            rate.sleep()

    ###################################################################
    ### functions
    ###################################################################
    def velocity_limitation(self, world_ux, world_uy):
        vec = np.array([world_ux, world_uy])
        vel_norm = np.linalg.norm(vec)
        if vel_norm > self._umax:
            vec = vec / vel_norm * self._umax
        return vec


if __name__ == "__main__":
    rospy.init_node("agent", anonymous=True)
    agent = Agent(MyQP)
    agent.spin()
