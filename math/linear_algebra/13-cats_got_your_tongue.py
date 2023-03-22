#!/usr/bin/env python3
import numpy as np
"""
Task 13. Cat's Got Your Tongue
"""


def np_cat(mat1, mat2, axis=0):
    """
    > Function that concatenates two matrices along a specific axis
    > mat1 and mat2 can be interpreted as numpy.ndarrays, and never empty
    """
    return np.concatenate((mat1, mat2), axis)
