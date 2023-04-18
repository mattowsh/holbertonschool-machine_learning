#!/usr/bin/env python3
"""
Task 1. Layers
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for neural network

        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function that the layer should use
    """

    # Set the weights. Use this line to implement He et. al initialization
    # for the layer weights:
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Create the layer:
    layer = tf.layers.Dense(units=n, name="layer", activation=activation,
                            kernel_initializer=weights)

    return layer(prev)
