#!/usr/bin/env python3
"""
Tasks: One-Hot Matrix
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a numeric label vector into a one-hot matrix

        one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
            classes: maximum number of classes
            m: number of examples
    """
    try:
        # Create a base np.array:
        decoded_matrix = np.zeros(shape=(one_hot.shape[1]))

        # Contemplate all classes:
        # In this tasks: from 0 to 9:
        classes = np.array(range(one_hot.shape[0]))

        # Calculate the one-hot decode vector for each class:
        pre_decode = (one_hot.T * classes).T

        # Sum each pre-decode row to the base np.array that we created:
        for i in pre_decode:
            decoded_matrix += i
        return decoded_matrix.astype(int)
    except Exception as e:
        return None
