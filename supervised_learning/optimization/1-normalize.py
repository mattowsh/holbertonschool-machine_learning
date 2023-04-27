#!/usr/bin/env python3
"""
Task 1. Normalize
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

        X: numpy.ndarray (d, nx) to normalize
            m: number of data points
            nx: number of features
        m: numpy.ndarray (nx,), contains the mean of all features of X
        s: numpy.ndarray (nx,), contains the standard deviation of all
        features of X
    """

    return ((X - m) / s)
