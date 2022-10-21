#!/usr/bin/env python
# -*- coding: utf-8 -*-

from angle_aware_control.field_generator import FieldGenerator

import rospy
import rospkg

import numpy as np
from matplotlib import pyplot as plt
import jax.numpy as jnp
from jax import jit, device_put

class PsiGenerator:
    def __init__(self):
        angle_aware_params = rospy.get_param("angle_aware")
        ref_z = rospy.get_param("agents/ref_z")
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("angle_aware_control")
        data_dir = pkg_path + "/data/input/"
        
        psi_param = angle_aware_params["psi"]
        phi_param = angle_aware_params["phi"]
        phi_generator = FieldGenerator(phi_param)
        psi_generator = FieldGenerator(psi_param)
        self._psi = psi_generator.generate_phi()
        self._psi_grid = psi_generator.generate_grid()
        self._psi_generator = psi_generator
        self._phi = phi_generator.generate_phi()
        self._phi_grid = phi_generator.generate_grid()
        self._phi_generator = psi_generator

        phi_grid = phi_generator.generate_grid()
        psi_linspace = psi_generator.get_linspace()
        phi = phi_generator.generate_phi()
        A = phi_generator.get_point_dense()
        psi_shape = psi_generator.get_shape()
        grid_span = psi_generator.get_grid_span()

        rospy.loginfo("phi : {}".format(phi.shape))
        rospy.loginfo("psi : {}".format(psi.shape))
        rospy.loginfo("compress start")
        start = rospy.Time.now()
        ###################################### compression
        phi_grid_jnp = device_put(phi_grid)
        zeta = self.zeta(phi_grid_jnp, ref_z)
        psi = self.compress(phi, zeta, psi_shape, psi_linspace, grid_span, A).block_until_ready()

        # for i, x in enumerate(psi_linspace[0]):
        #     for j, y in enumerate(psi_linspace[1]):
        #         A_k = (jnp.abs(x-zeta[0]) < grid_span[0]*0.5) * (jnp.abs(y-zeta[1]) < grid_span[1]*0.5)
        #         psi[i,j] = jnp.sum(A_k * phi) * A
        #######################################
        dt = rospy.Time.now() - start
        rospy.loginfo("calc time: {}".format(dt.to_sec()))

        save_name = data_dir + "psi"
        np.save(save_name, psi)
        plt.imshow(psi)
        plt.show()
        
    #############################################################
    # functions
    #############################################################
    @jit
    def zeta(self, grid, ref_z):
        common = (ref_z - grid[2]) * jnp.tan(jnp.pi*0.5 - grid[3])
        x = grid[0] - common * jnp.cos(grid[4])
        y = grid[1] - common * jnp.sin(grid[4])
        return x, y
    @jit
    def compress(self, phi, zeta, psi_shape, psi_linspace, grid_span, A):
        phi_jnp = device_put(phi)
        psi_jnp =jnp.zeros(psi_shape)
        for i, x in enumerate(psi_linspace[0]):
            for j, y in enumerate(psi_linspace[1]):
                psi_jnp[i,j] = jnp.sum((jnp.abs(x-zeta[0]) < grid_span[0]*0.5) * (jnp.abs(y-zeta[1]) < grid_span[1]*0.5) * phi_jnp) * A
        return psi_jnp

if __name__ == "__main__":
    rospy.init_node("central", anonymous=True)
    node = PsiGenerator()