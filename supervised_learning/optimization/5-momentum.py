#!/usr/bin/env python3
"""
Task 5. Momentum
"""
import numpy as n


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization
    algorithm

        alpha: learning rate
        beta: momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var
    """

    # Update the momentum vector:
    momentum = beta1 * v + (1 - beta1) * grad

    # Update the variable value:
    updated_value = var - (alpha * momentum)

    # Final return:
    return updated_value, momentum
