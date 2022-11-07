#!/usr/bin/env python
# -*- coding: utf-8 -*-

from coverage_util.field_generator import FieldGenerator

import rospy
import rospkg

import numpy as np
from matplotlib import pyplot as plt

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


def extract(x, y, zeta, psi_grid_span, phi, A):
    """_summary_

    Args:
        x (float): psi_x
        y (float): psi_y
        zeta (ndarray): [x_array, y_array]
        psi_grid_span (list): [dx, dy]
        phi (ndarray): importance map
        A (float): dq

    Returns:
        float: compressed importance of (x,y)
    """
    return (
        jnp.sum(
            (jnp.abs(x - zeta[0]) < psi_grid_span[0] * 0.5)
            * (jnp.abs(y - zeta[1]) < psi_grid_span[1] * 0.5)
            * phi
        )
        * A
    )


def project(zeta, psi_grid_span, psi_min):
    x_ids = jnp.round((zeta[0] - psi_min[0]) / psi_grid_span[0])
    y_ids = jnp.round((zeta[1] - psi_min[1]) / psi_grid_span[1])
    return x_ids, y_ids


def pick(phi, ids, x_id, y_id):
    return jnp.sum((ids[0] == x_id) * (ids[1] == y_id) * phi)


def compress2(phi, ids, psi_shape):
    psi = np.zeros(psi_shape)
    for x_id in range(psi_shape[0]):
        for y_id in range(psi_shape[1]):
            psi[x_id, y_id] = pick(phi, ids, x_id, y_id)
    return psi


class PsiGenerator:
    def __init__(self):
        angle_aware_params = rospy.get_param("angle_aware")
        ref_z = rospy.get_param("agents/ref_z")
        file_path = rospy.get_param("psi_path")
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("angle_aware_control")
        data_dir = pkg_path + "/data/input/"

        psi_param = angle_aware_params["psi"]
        phi_param = angle_aware_params["phi"]
        phi_generator = FieldGenerator(phi_param)
        psi_generator = FieldGenerator(psi_param)
        # self._psi = psi_generator.generate_phi()
        # self._psi_grid = psi_generator.generate_grid()
        # self._psi_generator = psi_generator
        # self._phi = phi_generator.generate_phi()
        # self._phi_grid = phi_generator.generate_grid()
        # self._phi_generator = psi_generator

        phi_grid = phi_generator.generate_grid(sparse=True)
        phi = phi_generator.generate_phi()
        A = phi_generator.get_point_dense()
        # psi_linspace = psi_generator.get_linspace()
        psi_grid_x, psi_grid_y = psi_generator.generate_grid(sparse=False)
        psi_shape = psi_generator.get_shape()
        psi_grid_span = psi_generator.get_grid_span()
        psi_limit = psi_generator.get_limit()

        psi_min = psi_limit[:, 0]
        # rospy.loginfo(phi)
        # rospy.loginfo("A : {}".format(A))
        rospy.loginfo("phi : {}".format(phi.shape))
        rospy.loginfo("psi : {}".format(psi_shape))

        zeta = zeta_func(phi_grid, ref_z)
        ids = project(zeta, psi_grid_span, psi_min)
        rospy.loginfo("compress start")
        start = rospy.Time.now()
        ###################################### compression

        psi = compress2(phi, ids, psi_shape)
        #### vmapを使うにはメモリが足りない
        # v_extract = vmap(
        #     extract,
        #     in_axes=(
        #         0,
        #         0,
        #         None,
        #         None,
        #         None,
        #         None,
        #     ),
        # )
        # psi = v_extract(psi_grid_x, psi_grid_y, zeta, psi_grid_span, phi, A)
        # psi = compress(psi, psi_grid_x_jnp, psi_grid_y_jnp, zeta, psi_grid_span, phi, A)

        #######################################
        dt = rospy.Time.now() - start
        rospy.loginfo("calc time: {}".format(dt.to_sec()))

        psi = psi.reshape(psi_shape)
        np.save(file_path, psi)
        plt.imshow(psi)
        plt.show()

    #############################################################
    # functions
    #############################################################


if __name__ == "__main__":
    rospy.init_node("central", anonymous=True)
    node = PsiGenerator()
