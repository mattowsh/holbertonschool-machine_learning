#!/usr/bin/env python3
"""
Tasks: One-Hot Encode
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

        Y: numpy.ndarray with shape (m,), contains numeric class labels
            m: number of examples
        classes: maximum number of classes found in Y
    """
    try:
        one_hot_matrix = np.zeros(shape=(classes, Y.shape[0]))
        for i, value in enumerate(Y):
            one_hot_matrix[value, i] = 1
        return one_hot_matrix
    except Exception as e:
        return None
