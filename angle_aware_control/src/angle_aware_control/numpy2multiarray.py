#!/usr/bin/env python
# -*- coding: utf-8 -*-

from std_msgs.msg import MultiArrayDimension
import numpy as np


def numpy2multiarray(multiarray_type, np_array):
    """Convert numpy.ndarray to multiarray"""
    multiarray = multiarray_type()
    multiarray.layout.dim = [
        MultiArrayDimension(
            "dim%d" % i, np_array.shape[i], np_array.shape[i] * np_array.dtype.itemsize
        )
        for i in range(np_array.ndim)
    ]
    multiarray.data = np_array.reshape(1, -1)[0].tolist()
    return multiarray


def multiarray2numpy(pytype, dtype, multiarray):
    """Convert multiarray to numpy.ndarray"""
    # dims = map(lambda x: x.size, multiarray.layout.dim)
    dims = [multiarray.layout.dim[i].size for i in range(len(multiarray.layout.dim))]
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)
