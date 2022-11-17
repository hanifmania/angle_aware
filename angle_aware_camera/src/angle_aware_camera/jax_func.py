#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import jit


@jit
def zeta4d(phi_grid, ref_z):
    common = (ref_z - phi_grid[2]) * np.tan(np.pi * 0.5 - phi_grid[3])
    x = phi_grid[0] + common * np.cos(phi_grid[4])
    y = phi_grid[1] + common * np.sin(phi_grid[4])

    theta_v = phi_grid[3]
    theta_h = phi_grid[4] + np.pi
    return x, y, theta_v, theta_h


@jit
def dist2(pos, pitch, yaw, zeta):
    return (
        (pos[0] - zeta[0]) ** 2
        + (pos[1] - zeta[1]) ** 2
        + (pitch - zeta[2]) ** 2
        + (yaw - zeta[3]) ** 2
    )


@jit
def nearest_dist(all_positions, pitches, yaws, zeta):
    min_dist = dist2(all_positions[0], pitches[0], yaws[0], zeta)
    for pos, pitch, yaw in zip(all_positions, pitches, yaws):
        dist = dist2(pos, pitch, yaw, zeta)
        new = min_dist > dist
        min_dist = dist * new + min_dist * ~new

    return min_dist


@jit
def performance_function(sigma, dist_power):
    """_summary_

    Args:
        sigma (float): _description_
        dist_power (ndarray): |p-zeta|^2

    Returns:
        ndarray: h(p, q)
    """
    return np.exp(-dist_power / (2 * sigma))


@jit
def calc_J(phi):
    return np.sum(phi)


@jit
def phi_equation(phi, delta_decrease, h, dt):
    phi -= delta_decrease * h * phi * dt
    # print(np.sum(self._delta_decrease * h_max * self._phi * self._dt))
    phi = (0 < phi) * phi  ## the minimum value is 0


@jit
def calc_phi(all_positions, pitches, yaws, zeta, sigma, phi, delta_decrease, dt):
    min_dist = nearest_dist(all_positions, pitches, yaws, zeta)
    h = performance_function(sigma, min_dist)
    ret = phi_equation(phi, delta_decrease, h, dt)
    return ret
