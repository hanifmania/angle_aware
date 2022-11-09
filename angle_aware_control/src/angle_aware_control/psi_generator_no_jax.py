#!/usr/bin/env python
# -*- coding: utf-8 -*-

from angle_aware_control.psi_generator_base import PsiGeneratorBase

import rospy
import numpy as jnp


def zeta_func(phi_grid, ref_z):
    """zeta

    Note: The equation in the thesis is not correct. This is the correct equation.

    Args:
        phi_grid (ndarray): _description_
        ref_z (float): _description_

    Returns:
        ndarray: _description_
    """
    common = (ref_z - phi_grid[2]) * jnp.tan(jnp.pi * 0.5 - phi_grid[3])
    x = phi_grid[0] + common * jnp.cos(phi_grid[4])
    y = phi_grid[1] + common * jnp.sin(phi_grid[4])
    return x, y


def project(zeta, psi_grid_span, psi_min):
    x_ids = jnp.round((zeta[0] - psi_min[0]) / psi_grid_span[0])
    y_ids = jnp.round((zeta[1] - psi_min[1]) / psi_grid_span[1])
    return x_ids, y_ids


def pick(ids, x_id, y_id):
    return jnp.sum((ids[0] == x_id) * (ids[1] == y_id))  # *phiはphi=1なので省略


def set_device_func(phi_grid, ref_z, psi_grid_span, psi_min):
    return phi_grid, ref_z, psi_grid_span, psi_min


if __name__ == "__main__":
    rospy.init_node("psi_generator", anonymous=True)
    node = PsiGeneratorBase()
    node.main(set_device_func, zeta_func, project, pick)
