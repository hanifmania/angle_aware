#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.qp import QPUtil, CBF2QP, ULimit
from bebop_hatanaka_base.cbf import FieldCBF, CollisionAvoidanceCBF
from angle_aware_control.angle_aware_cbf import AngleAwareCBF


class MyQP:
    def __init__(self, field, collision_distance, angle_aware_params):
        """_summary_

        Args:
            field (list): [[x_min x_max], [y_min y_max], [z_min z_max]]
            collision_distance (float):
        """
        self._qp_solver = QPUtil()
        # self._ulimit_qp = ULimit()
        self._cbf2qp = CBF2QP()
        self._field_cbf = FieldCBF()
        self._collision_avoidance_cbf = CollisionAvoidanceCBF()
        self._angle_aware_cbf = AngleAwareCBF()

        self._u_dim = 2
        self._slack_dim = 1
        costs = [1, 1, 1.0e6]  ### [ux, uy, angle_aware_slack]
        alpha_default = 1
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

    def solve(self, u_nom, pos, neighbor_pos, psi_grid, psi):
        """
        Args:
            u_nom (ndarray):
            pos (ndarray):
            neighbor_pos (ndarray):

        Returns:
            ndarray: u_optimal
            ndarray: slack
        """
        P_np, Q_np, G_np, h_np = self._cbf2qp.initPQGH(u_nom)

        ###### generate G, h from CBF
        ### field CBF
        field_dhdp, field_h = self._field_cbf.cbf(pos)
        # print(field_dhdp.shape, field_h.shape)
        G_np, h_np = self._cbf2qp.cbf2Gh(field_dhdp, field_h, G_np, h_np, slack_id=None)

        ### collision avoidance
        dhdps, alpha_hs = self._collision_avoidance_cbf.cbf(pos, neighbor_pos)
        for dhdp, alpha_h in zip(dhdps, alpha_hs):
            G_np, h_np = self._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=None)

        ### ulimit
        # G_np, h_np = self._ulimit_qp.Gh(G_np, h_np)

        ############ angle aware
        dhdp, alpha_h = self._angle_aware_cbf.cbf(pos, neighbor_pos, psi_grid, psi)
        G_np, h_np = self._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=0)

        ### solve
        u_optimal, solver_status = self._qp_solver.solve(P_np, Q_np, G_np, h_np)

        # print(dhdp, alpha_h, u_optimal)
        return u_optimal[: self._u_dim], u_optimal[self._u_dim :]
        return None, None
