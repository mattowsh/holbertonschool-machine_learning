#!/usr/bin/env python3
"""
Task 7. RMSProp
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: the previous second moment of var
    """

    momentum = beta2 * s + (1 - beta2) * (grad ** 2)
    updated_value = var - alpha * (grad / (np.sqrt(momentum) + epsilon))

    return updated_value, momentum
