#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase
from bebop_hatanaka_lab_example.myqp import MyQP

import rospy
import numpy as np


class Agent:
    def __init__(self):
        self.agentID = rospy.get_param("agentID", default=-1)
        agents_param = rospy.get_param("/agents")
        collision_distance = rospy.get_param("collision_distance")
        self._camera_deg = rospy.get_param("initial_pose/camera_deg")
        field_cbf = rospy.get_param("/field_cbf")

        self._clock = agents_param["agent_manager_clock"]
        self._umax = agents_param["u_max"]

        self._agent_base = AgentBase(self.agentID)
        self._qp = MyQP(field_cbf, collision_distance)

        self._agent_base.wait_pose_stamped()
        self._agent_base.publish_camera_control(self._camera_deg)

    ###################################################################
    ### main
    ###################################################################
    def main_control(self):
        my_position, my_orientation = self._agent_base.get_my_pose()
        # all_positions = self._agent_base.get_all_positions()
        # all_orientations = self._agent_base.get_all_orientations()
        neighbor_positions = self._agent_base.get_neighbor_positions()

        #### joy input
        # uh_x, uh_y, uh_z, uh_w, uh_camera = self._agent_base.get_uh()

        ##########################################
        #  generate ux,uy,uz. You can write any code here
        ##########################################
        world_ux = self._umax
        if self.agentID == 1:
            world_ux *= -1
        world_uy = 0
        world_uz = 0
        omega_z = 0
        ##########################################

        ### x y z field limitation and collision avoidance with CBF
        u_nom = np.array([world_ux, world_uy, world_uz])
        u_opt, w = self._qp.solve(u_nom, my_position, neighbor_positions)
        world_ux, world_uy, world_uz = u_opt
        world_ux, world_uy = self.velocity_limitation(world_ux, world_uy)
        self._agent_base.publish_command_from_world_vel(
            world_ux, world_uy, world_uz, omega_z
        )


        dist = np.sqrt((my_position[0] - neighbor_positions[0][0]) ** 2 + (my_position[1] - neighbor_positions[0][1]) ** 2)
        rospy.loginfo("dist : {:.3f}, ux : {:.3f}".format(dist, world_ux))
        # self.camera_control(uh_camera)

    ###################################################################
    ### spin
    ###################################################################
    def spin(self):
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            if self._agent_base.is_main_ok():
                self.main_control()
            rate.sleep()

    ###################################################################
    ### functions
    ###################################################################
    def camera_control(self, u_camera):
        """
        Args:
            u_camera (float): camera tilt deg/s
        """
        self._camera_deg += u_camera / self._clock
        self._camera_deg = np.clip(self._camera_deg, -90, 5)
        self._agent_base.publish_camera_control(self._camera_deg)

    def velocity_limitation(self, world_ux, world_uy):
        world_ux = np.clip(world_ux, -self._umax, self._umax)
        world_uy = np.clip(world_uy, -self._umax, self._umax)
        return world_ux, world_uy


if __name__ == "__main__":
    rospy.init_node("agent", anonymous=True)
    agent = Agent()
    agent.spin()
