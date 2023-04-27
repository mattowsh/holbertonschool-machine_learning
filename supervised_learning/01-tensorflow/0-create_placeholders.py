#!/usr/bin/env python3
"""
Task 0: Placeholders
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for the neural network

        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
    """

    # x is the placeholder for the input data to the neural network
    x = tf.placeholder("float32", name="x", shape=(None, nx))

    # y is the placeholder for the one-hot labels for the input data
    y = tf.placeholder("float32", name="y", shape=(None, classes))

    return x, y
