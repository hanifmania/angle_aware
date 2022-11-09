#!/usr/bin/env python
# -*- coding: utf-8 -*-
from coverage_util.voronoi import CoverageUtil

import numpy as np
from scipy.stats import norm


class AngleAwareCBF:
    def __init__(self):
        self._coverage_util = CoverageUtil()
        # self._J = 40401
        # self._J_dot_old = 0
        # self._b_i_old = 0

    def set_params(self, sigma, delta_decrease, gamma, alpha):
        self._sigma = sigma
        self._delta_decrease = delta_decrease
        self._alpha = alpha
        self._gamma = gamma

    def setU(self, ux, uy):
        """debug用

        Args:
            ux (_type_): _description_
            uy (_type_): _description_
        """
        self._ux = ux
        self._uy = uy

    def cbf(self, pos, neighbors, psi_grid, psi):
        """圧縮した重要度マップpsiを用いて、 -dot(J) - \gamma >=0を達成する

        Args:
            pos (ndarray): _description_
            neighbors (ndarray): _description_
            psi_grid (ndarray): _description_
            psi (ndarray): compressed importance

        Returns:
            ndarray: dhdp
            ndarray: alpha(h) + dhdpsi
        """
        region = self._coverage_util.calc_voronoi(pos, neighbors, psi_grid)
        performance_function = self.performance_function(pos, psi_grid)
        I_i_map = region * self._delta_decrease * performance_function * psi
        b_i = np.sum(I_i_map) - self._gamma
        print(b_i)
        temp = -1.0 / (self._sigma**2) * I_i_map
        db_dp_x = np.sum(temp * (pos[0] - psi_grid[0]))
        db_dp_y = np.sum(temp * (pos[1] - psi_grid[1]))

        dbdt = np.sum(-self._delta_decrease * performance_function * I_i_map)
        alpha_bi = self._alpha * b_i

        dhdp = np.array([[db_dp_x], [db_dp_y]])
        alpha_h = dbdt + alpha_bi

        ### debug用. 数値解と解析解が一致するか確認
        # J_new = np.sum(psi)
        # J_dot = (J_new - self._J)/0.05
        # self._J = J_new
        # print(J_dot)
        # J_ddot = (J_dot - self._J_dot_old) * 20
        # self._J_dot_old = J_dot

        # b_i_dot = (b_i - self._b_i_old)*20
        # self._b_i_old = b_i
        # db = db_dpsi + db_dp_x *self._ux #+ db_dp_y * self._uy
        # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(b_i_dot, db,  b_i_dot - db,db_dpsi, db_dp_x *self._ux ))

        # alpha_h = -10
        # print(dhdp, alpha_h)
        return dhdp, alpha_h

    def performance_function(self, pos, psi_grid):
        """_summary_

        Args:
            pos (ndarray): _description_
            psi_grid (ndarray): _description_

        Returns:
            ndarray: h(x - chi) map
        """
        dist_map = self._coverage_util.calc_dist(pos, psi_grid)
        h = norm.pdf(dist_map, scale=self._sigma) * np.sqrt(2 * np.pi) * self._sigma
        return h
