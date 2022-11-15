#!/usr/bin/env python
# -*- coding: utf-8 -*-
from angle_aware_control.myqp import MyQP


class AvoidObjectQP:
    def __init__(self, field, collision_distance, angle_aware_params, avoid_objects):
        self._myqp = MyQP(field, collision_distance, angle_aware_params)
        self._avoid_objects = avoid_objects

    def generate_PQGH(self, u_nom, pos, neighbor_pos, psi_grid, psi):
        P_np, Q_np, G_np, h_np = self._myqp.generate_PQGH(
            u_nom, pos, neighbor_pos, psi_grid, psi
        )
        dhdp, alpha_h = self._myqp._collision_avoidance_cbf.nearest_cbf(
            pos, self._avoid_objects["xy"], self._avoid_objects["r"]
        )
        G_np, h_np = self._myqp._cbf2qp.cbf2Gh(dhdp, alpha_h, G_np, h_np, slack_id=None)

    def solve(self, u_nom, pos, neighbor_pos, psi_grid, psi):
        P_np, Q_np, G_np, h_np = self.generate_PQGH(
            u_nom, pos, neighbor_pos, psi_grid, psi
        )
        ### solve
        u_optimal, solver_status = self._myqp._qp_solver.solve(P_np, Q_np, G_np, h_np)
        # print(G_np, h_np,u_optimal)

        # print(dhdp, alpha_h, u_optimal)
        return u_optimal[: self._u_dim], u_optimal[self._u_dim :]
