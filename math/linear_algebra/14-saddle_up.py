#!/usr/bin/env python3
"""
Task 14. Saddle Up
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    > Funtion that performs matrix multiplication
    > mat1 and mat2 are numpy.ndarrays and never empty
    """
    # matmul() gives us the matrix product of two 2-d arrays
    # Is great for times when we’re unsure of what the dimensions of our
    # matrices will be :)
    return np.matmul(mat1, mat2)
