#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase
from angle_aware_control.myqp import MyQP
from angle_aware_control.psi_generator_no_jax import zeta_func
from coverage_util.field_generator import FieldGenerator
from coverage_util.numpy2multiarray import numpy2multiarray
from coverage_util.voronoi import CoverageUtil

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, Float32MultiArray

import copy
import numpy as np

# from jsk_rviz_plugins.msg import Pictogram
# import matplotlib.pyplot as plt


class Agent:
    def __init__(self, myqp):
        ################## launch input topic name
        flag_topic = rospy.get_param("~flag_topic", default="angle_aware_mode")
        input_detect_topic = rospy.get_param(
            "~input_detect_topic", default="object_detector/target_posestamped"
        )
        output_J_topic = rospy.get_param("~output_J_topic", default="angle_aware/J")
        output_phi_topic = rospy.get_param(
            "~output_phi_topic", default="angle_aware/phi"
        )
        ################## yaml input
        angle_aware_params = rospy.get_param("~angle_aware", default=None)
        self.agentID = rospy.get_param("agentID", default=-1)
        collision_distance = rospy.get_param("collision_distance")
        ################## central input
        self._field_cbf = rospy.get_param("/field_cbf")
        agents_param = rospy.get_param("/agents")
        self._tree_params = rospy.get_param("/trees")

        self._clock = agents_param["agent_manager_clock"]
        self._kp_z = agents_param["kp_z"]
        self._kp_yaw = agents_param["kp_yaw"]
        self._ref_yaw = agents_param["ref_yaw"]
        self._unom_max = agents_param["unom_max"]
        self._umax = agents_param["u_max"]
        self._phi_param = angle_aware_params["phi"]
        self._delta_decrease = angle_aware_params["delta_decrease"]
        self._sigma = angle_aware_params["sigma"]
        # self._observe_time = angle_aware_params["observe_time"]
        self._ref_z = angle_aware_params["ref_z"]
        self._tau = angle_aware_params["tau"]

        phi_generator = FieldGenerator(self._phi_param)
        self._A = phi_generator.get_point_dense()

        phi_A = phi_generator.generate_phi() * self._A
        rospy.loginfo("J(0): {}".format(np.sum(phi_A)))

        self._phi_A = 0
        # self._phi_0 = 1.0
        self._object_queue = []
        self._dt = 1.0 / self._clock
        self._coverage_util = CoverageUtil()

        self._agent_base = AgentBase(self.agentID)
        self._qp = myqp(
            self._field_cbf,
            collision_distance,
            angle_aware_params,
            angle_aware_params["slack_cost"],
        )
        self._qp.set_obstacle_avoidance_param(self._tree_params)

        self._pub_flag = rospy.Publisher(flag_topic, Bool, queue_size=1)
        self._pub_J = rospy.Publisher(output_J_topic, Float32, queue_size=1)
        self._pub_phi = rospy.Publisher(
            output_phi_topic, Float32MultiArray, queue_size=1
        )
        rospy.Subscriber(input_detect_topic, PoseStamped, self.object_callback)
        self._agent_base.wait_pose_stamped()

    #############################################################
    # callback
    #############################################################
    def object_callback(self, msg):
        self._object_queue.append(msg)

    ###################################################################
    ### publish
    ###################################################################
    def publish_phi(self, phi):
        """_summary_

        Args:
            phi (ndarray): 重要度
        """
        multiarray = numpy2multiarray(Float32MultiArray, phi)
        self._pub_phi.publish(multiarray)

    ###################################################################
    ### main
    ###################################################################
    def main_control(self):
        my_position, my_orientation = self._agent_base.get_my_pose()
        neighbor_positions = self._agent_base.get_neighbor_positions()
        yaw = self._agent_base.get_my_yaw()
        all_positions = self._agent_base.get_all_positions()

        self._phi_A = self.update_phi(
            all_positions,
            self._zeta,
            self._sigma,
            self._delta_decrease,
            self._dt,
            self._phi_A,
        )
        self.publish_phi(self._phi_A)
        self._pub_J.publish(np.sum(self._phi_A))
        #### joy input
        # uh_x, uh_y, uh_z, uh_w, uh_camera = self._agent_base.get_uh()

        ##########################################
        #  generate ux,uy,uz. You can write any code here
        ##########################################
        # world_ux = 0  # uh_x
        # world_uy = 0  # uh_y

        #### unom = voronoi centeroid
        voronoi = self._coverage_util.calc_voronoi(
            my_position[:2], neighbor_positions[:, :2], self._zeta
        )

        temp = voronoi * self._phi_A
        mass = np.sum(temp)

        cent_x = 1.0 / mass * np.sum(temp * self._zeta[0])
        cent_y = 1.0 / mass * np.sum(temp * self._zeta[1])

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
            u_nom, my_position[:2], neighbor_positions[:, :2], self._zeta, self._phi_A
        )
        world_ux, world_uy = u_opt
        world_ux, world_uy = self.velocity_limitation(world_ux, world_uy, self._umax)
        self._agent_base.publish_command_from_world_vel(
            world_ux, world_uy, world_uz, omega_z
        )
        vel = np.linalg.norm([world_ux, world_uy])
        rospy.loginfo(
            "agent {}, |u|: {:.2f} ({:.2f}, {:.2f}), w: {:.6f}".format(
                self.agentID, vel, world_ux, world_uy, w[0]
            )
        )

    def judge_angle_aware(self):
        """angle awareかpatrolかを判断. また target filedの生成も行う

        Returns:
            bool: true if angle_aware
        """

        angle_aware_mode = np.sum(self._phi_A) > self._tau
        if angle_aware_mode:
            ### まだangle awareすべき
            return True
        if len(self._object_queue) == 0:
            ### もう見るべきぶどうが無い
            self._phi_A = 0
            return False

        ### target fieldを新しく生成
        self._object_posestamped = self._object_queue.pop()
        self._phi_A, self._zeta = self.generate_q(
            self._object_posestamped, self._phi_param, self._ref_z
        )
        self._phi_A = self.delete_tree_phi(
            self._phi_A,
            self._zeta,
            self._tree_params["xy"],
            self._tree_params["no_phi_radius"],
        )
        self._phi_A = self.bound_q_in_field(self._phi_A, self._zeta, self._field_cbf)
        # gamma = self._phi_0 / self._observe_time
        # rospy.loginfo("gamma : {}".format(gamma))
        # self._qp._angle_aware_cbf.set_gamma(gamma)
        return True

    ###################################################################
    ### spin
    ###################################################################
    def spin(self):
        # self._clock=1
        rate = rospy.Rate(self._clock)
        old_is_angle_aware = True
        while not rospy.is_shutdown():
            if self._agent_base.is_main_ok():
                is_angle_aware = self.judge_angle_aware()

                ### publish mode only when it changes
                if old_is_angle_aware != is_angle_aware:
                    self._pub_flag.publish(is_angle_aware)
                old_is_angle_aware = is_angle_aware
                if is_angle_aware:
                    self.main_control()
                    # self.show_pictogram()
            rate.sleep()

    ###################################################################
    ### functions
    ###################################################################

    def velocity_limitation(self, world_ux, world_uy, umax):
        """最大速度制限. ベクトルの方向は維持して大きさだけ変える

        Args:
            world_ux (float): _description_
            world_uy (float): _description_
            umax (float): _description_

        Returns:
            ndarray: [x, y]
        """
        vec = np.array([world_ux, world_uy])
        vel_norm = np.linalg.norm(vec)
        if vel_norm > umax:
            vec = vec / vel_norm * umax
        return vec

    def generate_q(self, posestamped, param_base, ref_z):
        """Qを生成

        Args:
            posestamped (Posestamped): 物体位置
            param_base (dict): phi_param
            ref_z (float): drone z for zeta

        Returns:
            ndarray: phi
            ndarray: zeta
        """
        x = posestamped.pose.position.x
        y = posestamped.pose.position.y
        z = posestamped.pose.position.z
        param = copy.deepcopy(param_base)
        range = np.array(param["range"])
        range[0] += x
        range[1] += y
        range[2] += z
        param["range"] = range

        generator = FieldGenerator(param)
        phi_A = generator.generate_phi() * self._A
        grid = generator.generate_grid()
        zeta = zeta_func(grid, ref_z)
        return phi_A, zeta

    def performance_function(self, pos, grid, sigma):
        """性能関数

        Args:
            pos (ndarray): [x,y]
            grid (ndarray): zeta
            sigma (float): _description_

        Returns:
            ndarray: h
        """

        dist2 = (pos[0] - grid[0]) ** 2 + (pos[1] - grid[1]) ** 2

        return np.exp(-dist2 / (2 * sigma**2))

    def update_phi(self, all_positions, grid, sigma, delta_decrease, dt, phi):
        """重要度更新

        Args:
            all_positions (ndarray): [[x,y]]
            grid (ndarray): zeta
            sigma (float): _description_
            delta_decrease (float): _description_
            dt (float): _description_
            phi (ndarray): 重要度

        Returns:
            ndarray: phi
        """
        all_positions = self._agent_base.get_all_positions()
        performance_functions = [
            self.performance_function(pos, grid, sigma) for pos in all_positions
        ]
        dist2 = np.stack(performance_functions)
        h_max = dist2.max(axis=0)
        phi -= delta_decrease * h_max * phi * dt
        # print(np.sum(self._delta_decrease * h_max * self._psi * self._dt))
        return (0 < phi) * phi  ## the minimum value is 0

    def delete_tree_phi(self, phi, grid, xy_list, radius_list):
        """木のあるところの重要度を消去

        Args:
            phi (ndarray): _description_
            grid (ndarray): zeta
            xy_list(list): 障害物位置[[x,y], ...]
            radius_list(list): 障害物半径 [r1, r2,...]

        Returns:
            ndarray: 木があるところを0にした重要度
        """
        ok = np.ones_like(phi)
        for xy, r in zip(xy_list, radius_list):
            dist = (grid[0] - xy[0]) ** 2 + (grid[1] - xy[1]) ** 2
            ok = ok * (dist > r**2)
        phi = phi * ok
        return phi

    def bound_q_in_field(self, phi, grid, field_param):
        """field CBF外の重要度を消去

        Args:
            phi (ndarray): _description_
            grid (ndarray): zeta
            field_param (dict): range=[[min_x, max_x], [min_y, max_y],...]

        Returns:
            ndarray: field内のみの重要度
        """
        range = np.array(field_param)
        ok = (
            (range[0][0] < grid[0])
            * (grid[0] < range[0][1])
            * (range[1][0] < grid[1])
            * (grid[1] < range[1][1])
        )
        phi = phi * ok
        return phi


if __name__ == "__main__":
    rospy.init_node("agent", anonymous=True)
    agent = Agent(MyQP)
    agent.spin()
