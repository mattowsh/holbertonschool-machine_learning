#!/usr/bin/env python3
"""
Task 9. Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm

        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: the previous first moment of var
        s: the previous second moment of var
        t: the time step used for bias correction
    """

    # Calculate the fst and snd moment estimates using the current gradient
    # and the previous estimates:
    fst_momentum = beta1 * v + (1 - beta1) * grad
    snd_momentum = beta2 * s + (1 - beta2) * (grad ** 2)

    # Calculate bias-correction for each momentum:
    fst_bias_correction = fst_momentum / (1 - (beta1 ** t))
    snd_bias_correction = snd_momentum / (1 - (beta2 ** t))

    # Update the variable:
    all_bias = snd_bias_correction / (np.sqrt(fst_bias_correction) + epsilon)
    updated_value = var - alpha * all_bias
