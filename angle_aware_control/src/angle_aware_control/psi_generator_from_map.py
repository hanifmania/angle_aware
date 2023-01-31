#!/usr/bin/env python
# -*- coding: utf-8 -*-

from angle_aware_control.psi_generator_base import PsiGeneratorBase
from angle_aware_control.psi_generator import (
    set_device_func,
    zeta_func,
    project,
    pick,
)
import rospy
import jax.numpy as jnp
import numpy as np
from jax import jit, device_put
import scipy
from PIL import Image


class PsiGeneratorFromMap(PsiGeneratorBase):
    def load_param(self):
        data_dir = rospy.get_param("~npy_data_dir")
        phi, phi_grid, ref_z, psi_grid_span, psi_min = super().load_param()
        file_name = "/../mat/IFAC2022.mat"
        file_path = data_dir + file_name
        importance_map = self.readFromMatlab(file_path)
        shape = phi.shape
        phi2d = self.resize(importance_map, shape[0], shape[1])
        ### normalize
        rospy.loginfo("importance max {}".format(phi2d.max()))
        phi2d = phi2d / phi2d.max()
        rospy.loginfo("normalized max {}".format(phi2d.max()))

        add_mesh = shape[2:]
        phi5d_hvzyx = self.image2phi(phi2d, add_mesh)
        phi5d = phi5d_hvzyx.T

        return phi5d, phi_grid, ref_z, psi_grid_span, psi_min

    def readFromMatlab(self, filepath):
        mat = scipy.io.loadmat(filepath)
        importance_map = mat["I"]
        return importance_map

    def resize(self, raw_map, mesh_acc_x, mesh_acc_y):
        image = Image.fromarray(raw_map)
        print("raw img size:", image.size)

        size = (mesh_acc_x, mesh_acc_y)
        resized_img = image.resize(size)
        # print(resized_img.size)
        resized_np = np.asarray(resized_img)  ### value = resized_np[y, x]
        return resized_np

    def image2phi(self, phi2d, virtual_add_mesh_acc):
        upsidedown_img = np.flipud(phi2d)  # 画像は左上が[0,0]。座標系は左下が原点なので、そちらに合わせる。
        temp = list(reversed(virtual_add_mesh_acc))
        tile_shape = temp + [1, 1]
        phi5d = np.tile(upsidedown_img, tile_shape)
        return phi5d


if __name__ == "__main__":
    rospy.init_node("psi_generator", anonymous=True)
    node = PsiGeneratorFromMap()
    node.main(set_device_func, zeta_func, project, pick)
