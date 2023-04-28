#!/usr/bin/env python3
"""
Task 14. Batch Normalization Upgraded
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function that should be used on the output of
        the layer
    """

    # Set general parameters:
    epsilon = 1e-8
    w_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Create a dense layer:
    new_layer = tf.layers.Dense(
        units=n,
        kernel_initializer=w_init
        )(prev)

    # Computes the mean and variance of a tensor along specified dimensions:
    # axes: the dimensions to reduce over to compute the mean and variance
    mean, var = tf.nn.moments(new_layer, axes=[0])

    # Set the trainable parameters:
    # Use tf.Variable(): function to create trainable variables in a
    # TensorFlow graph
    # Use tf.constant(): function to create a constant tensor in a
    # TensorFlow graph
    gamma = tf.Variable(
        tf.constant(1.0, shape=[n]),
        trainable=True,
        name="gamma"
    )

    beta = tf.Variable(
        tf.constant(0.0, shape=[n]),
        trainable=True,
        name="beta"
    )

    # Batch normalization:
    batch_norm_layer = tf.nn.batch_normalization(new_layer, mean, var,
                                                 beta, gamma, epsilon)

    # Return using the activation function:
    return (activation(batch_norm_layer))
