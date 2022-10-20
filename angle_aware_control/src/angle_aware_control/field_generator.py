#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class FieldGenerator:
    def __init__(self, param):
        '''
        param : range [[xmin xmax], [ymin ymax], ...]
                density: [x_dense, y_dense, ...]
        内部に実態を持つと余分なメモリを取るため、generatorとなっている
        '''
        self._param = param
        self._range_np = np.array(param["range"])
        self._density_np = np.array(param["density"])
        grid_shape = []
        linspaces = []
        for axes in range(len(self._density_np)) :
            range_min = self._range_np[axes][0]
            range_max = self._range_np[axes][1]
            density = self._density_np[axes]
            shape = int((range_max - range_min)/density + 1)
            linspace = np.linspace(range_min, range_max, shape)
            grid_shape.append(shape)
            linspaces.append(linspace)
        self._grid_shape = grid_shape
        self._linspaces = linspaces
    
    def generate_phi(self):
        ### shape はx,y,zの順にする. 過去のプログラム(task switch)とは逆なので注意
        return np.ones(self._grid_shape)  ### phi[x,y,z...]でアクセスできる
    
    def generate_grid(self, sparse=True):
        return np.meshgrid(*self._linspaces, indexing="ij", sparse=sparse) ### x_grid, y_grid, ...が得られる

    # def setPhi(self, phi):
    #     self._phi = phi

    # def getPhi(self, region=None):
    #     #### theta_h, theta_v, z, y, xの順
    #     if region is None:
    #         ret = self._phi
    #     else:
    #         ret = self._phi * region
    #     return ret

    # def getGrid(self, region=None):
    #     ##### xyだと左上が最も小さくなる。rvizに示すので注意

    #     if region is None:
    #         ret = self._grid
    #     else:
    #         ret = self._grid * region
    #     return ret

    def getGridSpan(self):
        return self._density_np

    def getLimit(self, axes):
        return self._range_np[axes]

    def getPointDense(self):
        return self._density_np.prod()

    # def getZeroIndex(self):
    #     #[TODO] not work now
    #     return self._zero_index
