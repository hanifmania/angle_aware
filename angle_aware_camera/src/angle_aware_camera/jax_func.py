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
def nearest_dist(all_positions, pitches, yaws, zeta, init_min_dist):

    min_dist = init_min_dist
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
