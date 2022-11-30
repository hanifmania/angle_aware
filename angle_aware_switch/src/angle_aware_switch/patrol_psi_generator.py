#!/usr/bin/env python
# -*- coding: utf-8 -*-

from coverage_util.field_generator import FieldGenerator

import rospy

import numpy as np


class PsiGeneratorBase:
    def __init__(self) -> None:
        data_dir = rospy.get_param("~npy_data_dir")
        angle_aware_params = rospy.get_param("angle_aware")
        psi_param = angle_aware_params["psi"]
        self._file_path = data_dir + psi_param["npy_name"]

        psi_generator = FieldGenerator(psi_param)
        psi = psi_generator.generate_phi()
        A = psi_generator.get_point_dense()
        psi = psi * A
        np.save(self._file_path, psi)


if __name__ == "__main__":
    rospy.init_node("psi_generator", anonymous=True)
    node = PsiGeneratorBase()
