#!/usr/bin/env python3
"""
Task 0. Create Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

        labels: one-hot numpy.ndarray (m, classes), contains the correct labels
        for each data point
            m : number of data points
            classes: number of classes
        logits: one-hot numpy.ndarray (m, classes), contains the predicted
        labels

    Returns: a confusion numpy.ndarray (classes, classes) with row indices
    representing the correct labels and column indices representing the
    predicted labels
    """

    return np.matmul(labels.T, logits)
