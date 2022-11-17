#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.qp import QPUtil, CBF2QP, ULimit
from bebop_hatanaka_base.cbf import FieldCBF, CollisionAvoidanceCBF
from angle_aware_camera.angle_aware_cbf import AngleAwareCBF

import numpy as np


class MyQP:
    def __init__(
        self, field, collision_distance, angle_aware_params, zeta, A, camera_limit
    ):
        self._qp_solver = QPUtil()
        # self._ulimit_qp = ULimit()
        self._cbf2qp = CBF2QP()
        self._field_cbf = FieldCBF()
        self._collision_avoidance_cbf = CollisionAvoidanceCBF()
        self._angle_aware_cbf = AngleAwareCBF()
        self._camera_cbf = FieldCBF()

        self._u_dim = 4
        self._slack_dim = 1
        costs = [1, 1, 1, 1, 1.0e4]  ### [ux, uy, angle_aware_slack]
        alpha_default = 5
        self._cbf2qp.set_dim(self._u_dim, self._slack_dim, costs)
        self._field_cbf.set_params(field, alpha_default)
        self._collision_avoidance_cbf.set_params(collision_distance, alpha_default)
        self._camera_cbf.set_params(camera_limit, alpha_default)
        # self._ulimit_qp.set_params(u_max, self._u_dim, self._slack_dim)

        ############ angle aware
        sigma = angle_aware_params["sigma"]
        delta_decrease = angle_aware_params["delta_decrease"]
        gamma = angle_aware_params["gamma"]
        alpha = angle_aware_params["alpha"]
        self._angle_aware_cbf.set_params(sigma, delta_decrease, gamma, alpha, zeta, A)

    def calc_PQGH(
        self,
        u_nom,
        pos,
        my_camera,
        my_yaw,
        neighbors,
        neighbors_camera,
        neighbors_yaw,
        phi,
    ):
        dhdp_zeros = np.zeros((self._u_dim, 1))
        P_np, Q_np, G_np, h_np = self._cbf2qp.initPQGH(u_nom)

        ###### generate G, h from CBF
        ### field CBF
        field_dhdp, field_h = self._field_cbf.cbf(pos)
        # print(field_dhdp, field_h)
        G_np, h_np = self._cbf2qp.cbf2Gh(field_dhdp, field_h, G_np, h_np, slack_id=None)

        ### collision avoidance
        dhdps, alpha_hs = self._collision_avoidance_cbf.cbf(pos, neighbors)
        if np.sum(np.array(alpha_hs) < 0):
            print("collision")
            # for pj in neighbor_pos:
            #     print(np.linalg.norm([pos[0]-pj[0], pos[1]-pj[1]]))
            print("dhdps", dhdps)
            print("alpha_hs", alpha_hs)
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
        dhdp, alpha_h = self._angle_aware_cbf.cbf(
            pos, my_camera, my_yaw, neighbors, neighbors_camera, neighbors_yaw, phi
        )
        # print(dhdp)
        G_np, h_np = self._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=0)

        #### camera limit
        # dhdp, field_h = self._camera_cbf.cbf(my_camera)

        # zero_dhdp = np.zeros((self._u_dim, 1))
        # zero_dhdp[2] = dhdp
        # G_np, h_np = self._cbf2qp.cbf2Gh(field_dhdp, field_h, G_np, h_np, slack_id=None)

        return P_np, Q_np, G_np, h_np

    def solve(self, P_np, Q_np, G_np, h_np):
        u_optimal, solver_status = self._qp_solver.solve(P_np, Q_np, G_np, h_np)
        return u_optimal[: self._u_dim], u_optimal[self._u_dim :]
