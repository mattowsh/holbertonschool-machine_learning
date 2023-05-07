#!/usr/bin/env python3
"""
Task 1. Gradient Descent with L2 Regularization
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization
    using Numpy and TensorFlow

    cost: a tensor containing the cost of the network without L2 regularization

    Returns: a tensor containing the cost of the network accounting for L2
    regularization
    """

    return cost + tf.losses.get_regularization_losses()
