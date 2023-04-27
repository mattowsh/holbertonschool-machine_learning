#!/usr/bin/env python3
"""
Task 5. Train_Op
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network

        loss: placeholder, the loss of the network's prediction
        alpha: placeholder, learning rate

        Returns: an operation that trains the network using gradient descent
    """

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    return optimizer.minimize(loss)
