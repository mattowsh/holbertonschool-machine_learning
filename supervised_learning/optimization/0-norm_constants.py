#!/usr/bin/env python3
"""
Task 0. Normalization Constants
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix

        X: numpy.ndarray of shape (m, nx) to normalize
            m: number of data points
            nx: number of features
    """

    # Calculates the mean and std deviation:
    return np.mean(X, axis=0), np.std(X, axis=0)
