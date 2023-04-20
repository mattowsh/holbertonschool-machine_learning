#!/usr/bin/env python3
"""
Task 4. Loss
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction

        y: a placeholder for the labels of the input data
        y_pred: a tensor containing the network's predictions
    """

    # Calculate softmax cross-entropy loss:
    ce_loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return ce_loss
