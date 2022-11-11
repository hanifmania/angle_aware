#!/usr/bin/env python
# -*- coding: utf-8 -*-

from coverage_util.field_generator import FieldGenerator

import rospy
import rospkg

import numpy as np
from matplotlib import pyplot as plt


class PsiGeneratorBase:
    def compress(self, pick_func, ids):
        psi_shape = self._psi_shape
        psi = np.zeros(psi_shape)
        for x_id in range(psi_shape[0]):
            for y_id in range(psi_shape[1]):
                psi[x_id, y_id] = pick_func(ids, x_id, y_id)
        return psi * self._A

    def main(self, set_device_func, zeta_func, project_func, pick_func):
        phi_grid, ref_z, psi_grid_span, psi_min = self.load_param()
        phi_grid, ref_z, psi_grid_span, psi_min = set_device_func(
            phi_grid, ref_z, psi_grid_span, psi_min
        )

        zeta = zeta_func(phi_grid, ref_z)
        ids = project_func(zeta, psi_grid_span, psi_min)
        rospy.loginfo("compress start")
        start = rospy.Time.now()
        psi = self.compress(pick_func, ids)

        dt = rospy.Time.now() - start
        rospy.loginfo("calc time: {}".format(dt.to_sec()))

        np.save(self._file_path, psi)
        plt.imshow(psi)
        plt.show()

    def load_param(self):
        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path("angle_aware_control")
        # data_dir = pkg_path + "/data/input/"
        data_dir = rospy.get_param("~npy_data_dir")
        angle_aware_params = rospy.get_param("angle_aware")
        ref_z = rospy.get_param("agents/ref_z")
        psi_param = angle_aware_params["psi"]
        phi_param = angle_aware_params["phi"]
        self._file_path = data_dir + psi_param["npy_name"]

        phi_generator = FieldGenerator(phi_param)
        psi_generator = FieldGenerator(psi_param)

        phi_grid = phi_generator.generate_grid(sparse=True)
        phi = phi_generator.generate_phi()
        self._A = phi_generator.get_point_dense()
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
        self._psi_shape = psi_shape
        return phi_grid, ref_z, psi_grid_span, psi_min
