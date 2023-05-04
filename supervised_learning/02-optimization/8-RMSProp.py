#!/usr/bin/env python3
"""
Task 8. RMSProp Upgraded
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm

        loss: loss of the network
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero
    """

    # Define the optimizer using gradient descent + RMSProp algorithm:
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                          epsilon=epsilon)

    # Use minimize: used to update the trainable variables of a neural network
    # by minimizing a given loss function using the Gradient Descent with
    # Momentum optimization algorithm
    return optimizer.minimize(loss)
