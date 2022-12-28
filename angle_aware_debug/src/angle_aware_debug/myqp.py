#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.qp import QPUtil, CBF2QP
from bebop_hatanaka_base.cbf import FieldCBF, CollisionAvoidanceCBF

class MyQP:
    def __init__(self, field, collision_distance):
        """_summary_

        Args:
            field (list): [[x_min x_max], [y_min y_max], [z_min z_max]]
            collision_distance (float): 
        """        
        self._qp_solver = QPUtil()
        self._cbf2qp = CBF2QP()
        self._field_cbf = FieldCBF()
        self._collision_avoidance_cbf = CollisionAvoidanceCBF()

        self._u_dim = 2
        self._slack_dim = 1
        costs = [1, 1, 1.0e+6] ##ux uy uz field_slack
        alpha = 0.5
        self._cbf2qp.set_dim(self._u_dim, self._slack_dim, costs)
        self._field_cbf.set_params(field, alpha)
        self._collision_avoidance_cbf.set_params(collision_distance, alpha)
    
    def solve(self, u_nom, pos, neighbor_pos):
        """
        Args:
            u_nom (ndarray): 
            pos (ndarray): 
            neighbor_pos (ndarray): 

        Returns:
            ndarray: [ux, uy, uz]
            ndarray: [slack]
        """        
        P_np, Q_np, G_np, h_np = self._cbf2qp.initPQGH(u_nom)

        ### generate G, h from CBF
        field_dhdp, field_h = self._field_cbf.cbf(pos)
        G_np, h_np = self._cbf2qp.cbf2Gh(field_dhdp, field_h, G_np, h_np, slack_id=0)

        dhdps, alpha_hs = self._collision_avoidance_cbf.cbf(pos, neighbor_pos)
        for dhdp, alpha_h in zip(dhdps, alpha_hs):
            G_np, h_np = self._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=None)


        ### solve
        u_optimal, solver_status = self._qp_solver.solve(P_np, Q_np, G_np, h_np)
        return u_optimal[:self._u_dim], u_optimal[self._u_dim:]

