#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase
from angle_aware_control.myqp import MyQP
from angle_aware_control.psi_generator_no_jax import zeta_func
from coverage_util.field_generator import FieldGenerator

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import copy


class Agent:
    def __init__(self, myqp):
        flag_topic = rospy.get_param("~flag_topic", default="angle_aware_mode")
        input_detect_topic = rospy.get_param(
            "~input_detect_topic", default="grape_posestamped"
        )
        self._grape_size = rospy.get_param("/grape_detector/size", default=None)
        self.agentID = rospy.get_param("agentID", default=-1)
        field_cbf = rospy.get_param("/field_cbf")
        agents_param = rospy.get_param("/agents")
        angle_aware_params = rospy.get_param("~angle_aware", default=None)
        self._clock = agents_param["agent_manager_clock"]
        self._umax = agents_param["u_max"]
        collision_distance = rospy.get_param("collision_distance")
        self._kp_z = agents_param["kp_z"]
        self._ref_z = agents_param["ref_z"]
        self._kp_yaw = agents_param["kp_yaw"]
        self._ref_yaw = agents_param["ref_yaw"]
        self._phi_param = angle_aware_params["phi"]
        self._threshold = angle_aware_params["threshold"]
        self._delta_decrease = angle_aware_params["delta_decrease"]
        self._sigma = angle_aware_params["sigma"]

        phi_generator = FieldGenerator(self._phi_param)
        self._A = phi_generator.get_point_dense()

        self._phi_A = 0
        self._grape_queue = []
        self._dt = 1.0 / self._clock

        self._agent_base = AgentBase(self.agentID)
        self._qp = myqp(field_cbf, collision_distance, angle_aware_params)

        self._pub_flag = rospy.Publisher(flag_topic, Bool, queue_size=1)
        rospy.Subscriber(input_detect_topic, PoseStamped, self.grape_callback)

        self._agent_base.wait_pose_stamped()
        # self._agent_base.publish_camera_control(self._camera_deg)

    #############################################################
    # callback
    #############################################################
    def grape_callback(self, msg):
        self._grape_queue.append(msg)

    ###################################################################
    ### main
    ###################################################################
    def main_control(self):
        my_position, my_orientation = self._agent_base.get_my_pose()
        neighbor_positions = self._agent_base.get_neighbor_positions()
        yaw = self._agent_base.get_my_yaw()

        self._phi_A = self.update_phi(
            my_position,
            self._zeta,
            self._sigma,
            self._delta_decrease,
            self._dt,
            self._phi_A,
        )
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
            u_nom, my_position[:2], neighbor_positions[:, :2], self._zeta, self._phi_A
        )
        world_ux, world_uy = u_opt
        world_ux, world_uy = self.velocity_limitation(world_ux, world_uy, self._umax)
        self._agent_base.publish_command_from_world_vel(
            world_ux, world_uy, world_uz, omega_z
        )
        vel = np.linalg.norm([world_ux, world_uy])
        rospy.loginfo(
            "agent {}, |u|: {:.2f} ({:.2f}, {:.2f}), w: {:.3f}, phi:{}".format(
                self.agentID, vel, world_ux, world_uy, w[0], np.sum(self._phi_A)
            )
        )

    ###################################################################
    ### spin
    ###################################################################
    def spin(self):
        # self._clock=1
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            if self._agent_base.is_main_ok():
                is_angle_aware = self.judge_angle_aware()
                self._pub_flag.publish(is_angle_aware)
                if is_angle_aware:
                    self.main_control()
            rate.sleep()

    ###################################################################
    ### functions
    ###################################################################
    def velocity_limitation(self, world_ux, world_uy, umax):
        vec = np.array([world_ux, world_uy])
        vel_norm = np.linalg.norm(vec)
        if vel_norm > umax:
            vec = vec / vel_norm * umax
        return vec

    def judge_angle_aware(self):
        """angle awareかpatrolかを判断. また target filedの生成も行う

        Returns:
            bool: true if angle_aware
        """

        angle_aware_mode = np.sum(self._phi_A) > self._threshold
        if angle_aware_mode:
            ### まだangle awareすべき
            return True
        if len(self._grape_queue) == 0:
            ### もう見るべきぶどうが無い
            self._phi_A = 0
            return False

        ### target fieldを新しくして再検証
        grape_posestamped = self._grape_queue.pop()
        self._phi_A, self._zeta = self.generate_q(
            grape_posestamped, self._phi_param, self._grape_size, self._ref_z
        )

        rospy.loginfo("max J : {}".format(np.sum(self._phi_A)))
        return self.judge_angle_aware()

    def generate_q(self, posestamped, param_base, grape_size, ref_z):
        x = posestamped.pose.position.x
        y = posestamped.pose.position.y
        z = posestamped.pose.position.z
        param = copy.deepcopy(param_base)
        range = param["range"]
        range[0] = [x - grape_size[0] * 0.5, x + grape_size[0] * 0.5]
        range[1] = [y - grape_size[1] * 0.5, y + grape_size[1] * 0.5]
        range[2] = [z - grape_size[2] * 0.5, z + grape_size[2] * 0.5]

        param["range"] = range
        generator = FieldGenerator(param)
        phi_A = generator.generate_phi() * self._A
        grid = generator.generate_grid()
        zeta = zeta_func(grid, ref_z)
        return phi_A, zeta

    def performance_function(self, pos, grid, sigma):

        dist2 = (pos[0] - grid[0]) ** 2 + (pos[1] - grid[1]) ** 2

        return np.exp(-dist2 / (2 * sigma**2))

    def update_phi(self, pos, grid, sigma, delta_decrease, dt, phi):
        h = self.performance_function(pos, grid, sigma)
        phi -= delta_decrease * h * phi * dt
        # print(np.sum(self._delta_decrease * h_max * self._psi * self._dt))
        return (0 < phi) * phi  ## the minimum value is 0


if __name__ == "__main__":
    rospy.init_node("agent", anonymous=True)
    agent = Agent(MyQP)
    agent.spin()
