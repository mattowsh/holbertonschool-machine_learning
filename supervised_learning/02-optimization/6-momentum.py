#!/usr/bin/env python3
"""
Task 6. Momentum Upgraded
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm

        loss: loss of the network
        alpha: learning rate
        beta1: momentum weight
    """

    # Define the optimizer using gradient descent + momentum:
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)

    # Use minimize: used to update the trainable variables of a neural network
    # by minimizing a given loss function using the Gradient Descent with
    # Momentum optimization algorithm
    return optimizer.minimize(loss)
