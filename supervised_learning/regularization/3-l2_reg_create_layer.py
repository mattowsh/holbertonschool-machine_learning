#!/usr/bin/env python3
"""
Task 3. Create a Layer with L2 Regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization

    prev: a tensor containing the output of the previous layer
    n: the number of nodes the new layer should contain
    activation: the activation function that should be used on the layer
    lambtha: L2 regularization parameter

    Returns: the output of the new layer
    """

    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    new_layer = tf.layers.dense(prev, n, activation=activation,
                                kernel_regularizer=regularizer)

    return new_layer
