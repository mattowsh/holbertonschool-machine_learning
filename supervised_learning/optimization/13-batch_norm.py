#!/usr/bin/env python3
"""
Task 13. Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization

        Z: numpy.ndarray (m, n) that should be normalized
            m: number of data points
            n: number of features in Z
        gamma: numpy.ndarray (1, n) containing the scales used for batch
        normalization
        beta: numpy.ndarray (1, n) containing the offsets used for batch
        normalization
        epsilon: small number used to avoid division by zero
    """

    # Calculate mean and variance:
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    # Normalize Z:
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)

    # Scale and shift Z normalizated:
    result = gamma * Z_norm + beta

    return result
