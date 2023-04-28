#!/usr/bin/env python3
"""
Task 10. Adam Upgraded
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm

        loss: loss of the network
        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero
    """

    # Define the optimizer using gradient descent + Adam algorithm:
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)

    # Use minimize: used to update the trainable variables of a neural network
    # by minimizing a given loss function using the Gradient Descent with
    # Momentum optimization algorithm
    return optimizer.minimize(loss)
