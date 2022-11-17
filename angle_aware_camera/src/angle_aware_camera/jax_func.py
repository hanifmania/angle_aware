#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import jit, device_put
from jax.debug import print


@jit
def rad_in_pi(val):
    """角度を[-pi. pi]に揃える

    Args:
        val (ndarray): [-2pi, 2pi]

    Returns:
        ndarray: [-pi, pi]
    """
    val = np.where(val > np.pi, val - 2 * np.pi, val)
    val = np.where(val < -np.pi, val + 2 * np.pi, val)
    return val


@jit
def zeta4d(phi_grid, ref_z):
    common = (ref_z - phi_grid[2]) * np.tan(np.pi * 0.5 - phi_grid[3])
    x = phi_grid[0] + common * np.cos(phi_grid[4])
    y = phi_grid[1] + common * np.sin(phi_grid[4])

    theta_v = -phi_grid[3]
    theta_h = phi_grid[4] + np.pi
    theta_h = rad_in_pi(theta_h)
    return x, y, theta_v, theta_h


@jit
def dist2(pos, camera, yaw, zeta):

    theta_h = yaw - zeta[3]
    theta_h = rad_in_pi(theta_h)

    return (
        (pos[0] - zeta[0]) ** 2
        + (pos[1] - zeta[1]) ** 2
        + (camera - zeta[2]) ** 2
        + theta_h**2
    )


@jit
def nearest_dist(all_positions, cameras, yaws, zeta):
    min_dist = np.ones_like(zeta[0]) * 1e3
    #  dist2(all_positions[0], cameras[0], yaws[0], zeta)
    for pos, camera, yaw in zip(all_positions, cameras, yaws):
        dist = dist2(pos, camera, yaw, zeta)
        min_dist = np.minimum(dist, min_dist)

    return min_dist


@jit
def performance_function(dist_power, sigma):
    """_summary_

    Args:
        sigma (float): _description_
        dist_power (ndarray): |p-zeta|^2

    Returns:
        ndarray: h(p, q)
    """
    return np.exp(-dist_power / (2 * sigma**2))


@jit
def calc_J(phi):
    return np.sum(phi)


@jit
def phi_equation(phi, delta_decrease, h, dt):
    phi -= delta_decrease * h * phi * dt
    return np.maximum(0, phi)  ## the minimum value is 0


@jit
def calc_phi(all_positions, cameras, yaws, zeta, sigma, phi, delta_decrease, dt):

    min_dist = nearest_dist(all_positions, cameras, yaws, zeta)
    h = performance_function(min_dist, sigma)
    ret = phi_equation(phi, delta_decrease, h, dt)
    return ret


@jit
def calc_voronoi(
    pos,
    my_camera,
    my_yaw,
    neighbors,
    neighbors_camera,
    neighbors_yaw,
    zeta,
):
    """_summary_

    Args:
        pos (ndarray): _description_
        neighbors (ndarray): _description_
        grid2d (ndarray): _description_

    Returns:
        ndarray: Voronoi = 1, neighbor = 0 map
    """
    my_dist = dist2(pos, my_camera, my_yaw, zeta)
    region = np.ones_like(my_dist)
    for neighbor, camera, yaw in zip(neighbors, neighbors_camera, neighbors_yaw):
        neighbor_dist = dist2(neighbor, camera, yaw, zeta)
        region = region * (my_dist < neighbor_dist)
    return region


@jit
def angle_aware_cbf(
    pos,
    my_camera,
    my_yaw,
    neighbors,
    neighbors_camera,
    neighbors_yaw,
    zeta,
    phi,
    delta_decrease,
    gamma,
    sigma,
    alpha,
    A,
):
    """圧縮した重要度マップphiを用いて、 -dot(J) - \gamma >=0を達成する

    Args:
        pos (ndarray): _description_
        neighbors (ndarray): _description_
        zeta (ndarray): _description_
        phi (ndarray): compressed importance

    Returns:
        ndarray: dhdp
        ndarray: alpha(h) + dhdphi
    """
    region = calc_voronoi(
        pos,
        my_camera,
        my_yaw,
        neighbors,
        neighbors_camera,
        neighbors_yaw,
        zeta,
    )
    dist_power = dist2(pos, my_camera, my_yaw, zeta)
    h = performance_function(dist_power, sigma)
    I_i_map = region * delta_decrease * h * phi * A
    b_i = np.sum(I_i_map) - gamma
    temp = -1.0 / (sigma**2) * I_i_map
    db_dp_x = np.sum(temp * (pos[0] - zeta[0]))
    db_dp_y = np.sum(temp * (pos[1] - zeta[1]))
    db_dp_camera = np.sum(temp * (my_camera - zeta[2]))

    theta_h = my_yaw - zeta[3]
    theta_h = rad_in_pi(theta_h)

    db_dp_omega = np.sum(temp * theta_h)
    dbdt = np.sum(-delta_decrease * h * I_i_map)
    alpha_bi = alpha * b_i

    dhdp = np.array([[db_dp_x], [db_dp_y], [db_dp_camera], [db_dp_omega]])
    alpha_h = dbdt + alpha_bi
    return dhdp, alpha_h
