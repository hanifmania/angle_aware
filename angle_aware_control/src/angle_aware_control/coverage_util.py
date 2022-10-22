#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class CoverageUtil:
    def calc_voronoi(self, pos, neighbors, grid2d):
        """_summary_

        Args:
            pos (ndarray): _description_
            neighbors (ndarray): _description_
            grid2d (ndarray): _description_

        Returns:
            ndarray: Voronoi = 1, neighbor = 0 map
        """
        my_dist = self.calc_dist(pos, grid2d)
        region = np.ones_like(my_dist)
        for neighbor in neighbors:
            neighbor_dist = self.calc_dist(neighbor, grid2d)
            region = region * (my_dist < neighbor_dist)
        return region

    def calc_dist(self, pos, grid2d):
        """_summary_

        Args:
            pos (ndarray): _description_
            grid2d (ndarray): _description_

        Returns:
            ndarray: dist map
        """
        dist = np.sqrt((pos[0] - grid2d[0]) ** 2 + (pos[1] - grid2d[1]) ** 2)
        return dist
