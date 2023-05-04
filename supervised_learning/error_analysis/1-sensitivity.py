#!/usr/bin/env python3
"""
Task 1. Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

        confusion: confusion numpy.ndarray (classes, classes)
            classes: number of classes

    Return: a numpy.ndarray (classes,) containing the sensitivity of each class
    """

    # sensibility == recall (RC) == true positive ratio (TPR):
    # the number of correct positive predictions (TP) divided by the total
    # number of positives (P) => TP / P == TP / (TP + FN)

    id_matrix = np.identity(confusion.shape[1])

    # Get all TRUE positive values (TP). These are in the main diagonal of
    # (confusion * id_matrix), non-diagonal values are zeros:
    # We obtain a 1D vector result of the sum along the rows:
    TP = np.sum(confusion * id_matrix, axis=1)

    # Get ALL positives (P), adding all values in confusion:
    # We obtain a 1D vector result of the sum along the rows:
    P = np.sum(confusion, axis=1)

    # Compute the recall value:
    return TP / P
