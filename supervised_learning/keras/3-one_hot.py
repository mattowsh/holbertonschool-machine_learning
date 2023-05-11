#!/usr/bin/env python3
"""
Task 2. Optimize
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix

        labels: vector that contains the labels to convert into one-hot matrix
        classes: classes for one-hot matrix

    Returns: the one-hot matrix
    """

    return K.utils.to_categorical(labels, num_classes=classes)
