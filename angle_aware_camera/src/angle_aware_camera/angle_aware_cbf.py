#!/usr/bin/env python
# -*- coding: utf-8 -*-

from angle_aware_camera.jax_func import angle_aware_cbf

from jax import device_put


class AngleAwareCBF:
    def set_params(self, sigma, delta_decrease, gamma, alpha, zeta, A):
        self._sigma = device_put(sigma)
        self._delta_decrease = device_put(delta_decrease)
        self._alpha = device_put(alpha)
        self._gamma = device_put(gamma)
        self._zeta = device_put(zeta)
        self._A = device_put(A)

    def cbf(
        self, pos, my_camera, my_yaw, neighbors, neighrbors_camera, neighbors_yaw, phi
    ):
        dhdp, alpha_h = angle_aware_cbf(
            pos,
            my_camera,
            my_yaw,
            neighbors,
            neighrbors_camera,
            neighbors_yaw,
            self._zeta,
            phi,
            self._delta_decrease,
            self._gamma,
            self._sigma,
            self._alpha,
            self._A,
        )
        return dhdp, alpha_h
