#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.qp import QPUtil, CBF2QP
from bebop_hatanaka_base.cbf import FieldCBF, CollisionAvoidanceCBF
from angle_aware_control.angle_aware_cbf import AngleAwareCBF

import numpy as np


class MyQP:
    def __init__(self, field, collision_distance, angle_aware_params):
        """_summary_

        Args:
            field (list): [[x_min x_max], [y_min y_max], [z_min z_max]]
            collision_distance (float):
        """
        self._qp_solver = QPUtil()
        self._cbf2qp = CBF2QP()
        self._field_cbf = FieldCBF()
        self._collision_avoidance_cbf = CollisionAvoidanceCBF()
        self._angle_aware_cbf = AngleAwareCBF()

        self._u_dim = 2
        self._slack_dim = 1
        if "slack_cost" in angle_aware_params:
            slack_cost= angle_aware_params["slack_cost"]
        else:
            slack_cost = 1.0e6
        costs = [1, 1, slack_cost]  ### [ux, uy, angle_aware_slack]
        alpha_default = 0.5
        self._cbf2qp.set_dim(self._u_dim, self._slack_dim, costs)
        self._field_cbf.set_params(field, alpha_default)
        self._collision_avoidance_cbf.set_params(collision_distance, alpha_default)
        # self._ulimit_qp.set_params(u_max, self._u_dim, self._slack_dim)

        ############ angle aware
        sigma = angle_aware_params["sigma"]
        delta_decrease = angle_aware_params["delta_decrease"]
        gamma = angle_aware_params["gamma"]
        alpha = angle_aware_params["alpha"]
        self._angle_aware_cbf.set_params(sigma, delta_decrease, gamma, alpha)

        self._is_obstacle_avoidance = False

    def set_obstacle_avoidance_param(self, obstacle_avoidance_param):
        self._is_obstacle_avoidance = True
        self._obstacle_avoidance_param = obstacle_avoidance_param

    def calc_PQGh(self, u_nom, pos, neighbor_pos, psi_grid, psi):
        P_np, Q_np, G_np, h_np = self._cbf2qp.initPQGH(u_nom)

        ###### generate G, h from CBF
        ### field CBF
        field_dhdp, field_h = self._field_cbf.cbf(pos)
        # print(field_dhdp, field_h)
        G_np, h_np = self._cbf2qp.cbf2Gh(field_dhdp, field_h, G_np, h_np, slack_id=None)

        ### collision avoidance
        dhdps, alpha_hs = self._collision_avoidance_cbf.cbf(pos, neighbor_pos)
        if np.sum(np.array(alpha_hs) < 0):
            print("collision")
            for pj in neighbor_pos:
                print(np.linalg.norm([pos[0] - pj[0], pos[1] - pj[1]]))
            # print("dhdps", dhdps)
            # print("alpha_hs", alpha_hs)
        for dhdp, alpha_h in zip(dhdps, alpha_hs):
            G_np, h_np = self._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=None)
        # dhdp, alpha_h = self._collision_avoidance_cbf.nearest_cbf(pos, neighbor_pos)
        # if alpha_h < 0:
        #     print("collision")
        # print(dhdp, alpha_h)

        # G_np, h_np = self._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=None)

        ### ulimit
        # G_np, h_np = self._ulimit_qp.Gh(G_np, h_np)

        ############ angle aware
        dhdp, alpha_h = self._angle_aware_cbf.cbf(pos, neighbor_pos, psi_grid, psi)
        G_np, h_np = self._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=0)

        ############ obstacle avoidance
        if self._is_obstacle_avoidance:
            dhdp, alpha_h = self._collision_avoidance_cbf.nearest_cbf(
                pos,
                self._obstacle_avoidance_param["xy"],
                self._obstacle_avoidance_param["avoid_radius"],
            )
            G_np, h_np = self._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=None)

        return P_np, Q_np, G_np, h_np

    def solve(self, u_nom, pos, neighbor_pos, psi_grid, psi):
        P_np, Q_np, G_np, h_np = self.calc_PQGh(u_nom, pos, neighbor_pos, psi_grid, psi)
        u_optimal, solver_status = self._qp_solver.solve(P_np, Q_np, G_np, h_np)
        return u_optimal[: self._u_dim], u_optimal[self._u_dim :]
