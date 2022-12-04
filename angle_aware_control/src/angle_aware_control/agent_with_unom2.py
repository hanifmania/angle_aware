#!/usr/bin/env python
# -*- coding: utf-8 -*-
from angle_aware_control.agent import Agent
from angle_aware_control.myqp import MyQP

from coverage_util.voronoi import CoverageUtil


import rospy

import numpy as np


class AgentWithUnom(Agent):
    def __init__(self, myqp):
        super(AgentWithUnom, self).__init__(myqp)
        self._unom_max = rospy.get_param("/agents/unom_max")
        self._coverage_util = CoverageUtil()

        ###################################################################
        ### main
        ###################################################################def main_control(self, psi=None):

    def main_control(self, psi=None):
        if psi is None:
            psi = self._psi
        my_position, my_orientation = self._agent_base.get_my_pose()
        neighbor_positions = self._agent_base.get_neighbor_positions()
        yaw = self._agent_base.get_my_yaw()
        #### joy input
        # uh_x, uh_y, uh_z, uh_w, uh_camera = self._agent_base.get_uh()

        ##########################################
        #  generate ux,uy,uz. You can write any code here
        ##########################################
        ## u_nom = [0,0]
        # world_ux = 0  # uh_x
        # world_uy = 0  # uh_y

        voronoi = self._coverage_util.calc_voronoi(
            my_position[:2], neighbor_positions[:, :2], self._psi_grid
        )

        temp = voronoi * self._psi
        mass = np.sum(temp)

        cent_x = 1.0 / mass * np.sum(temp * self._psi_grid[0])
        cent_y = 1.0 / mass * np.sum(temp * self._psi_grid[1])

        world_ux = cent_x - my_position[0]  # uh_x
        world_uy = cent_y - my_position[1]  # uh_y
        world_ux, world_uy = self.velocity_limitation(
            world_ux, world_uy, self._unom_max
        )
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
            u_nom, my_position[:2], neighbor_positions[:, :2], self._psi_grid, psi
        )
        world_ux, world_uy = u_opt
        world_ux, world_uy = self.velocity_limitation(world_ux, world_uy, self._umax)

        self._agent_base.publish_command_from_world_vel(
            world_ux, world_uy, world_uz, omega_z
        )
        vel = np.linalg.norm([world_ux, world_uy])
        rospy.loginfo(
            "agent {}, |u|: {:.2f} ({:.2f}, {:.2f}), w: {:.3f}".format(
                self.agentID, vel, world_ux, world_uy, w[0]
            )
        )


if __name__ == "__main__":
    rospy.init_node("agent", anonymous=True)
    node = AgentWithUnom(MyQP)
    node.spin()
