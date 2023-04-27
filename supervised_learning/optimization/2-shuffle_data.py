#!/usr/bin/env python3
"""
Task 2. Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

        X: numpy.ndarray of shape (m, nx) to shuffle
            m: number of data points
            nx: number of features in X
        X: numpy.ndarray of shape (m, ny) to shuffle
            m: same number of data points as in X
            ny: number of features in Y
    """

    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]
