#!/usr/bin/env python3
"""
Task 4. Loss
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction

        y: a placeholder for the labels of the input data
        y_pred: a tensor containing the network's predictions

        Returns: a tensor containing the loss of the prediction
    """

    # Calculate softmax cross-entropy loss:
    cross_e = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(cross_e)
