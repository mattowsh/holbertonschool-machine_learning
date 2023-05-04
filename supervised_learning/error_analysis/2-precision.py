#!/usr/bin/env python3
"""
Task 2. Precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

        confusion: confusion numpy.ndarray (classes, classes)
            classes: number of classes

    Returns: a numpy.ndarray (classes,) containing the precision of each class
    """

    # specifity (SP) == true negative rate (TNR)
    # number of correct negative predictions (TN) divided by the total number
    # of negatives (N) => TN / N

    id_matrix = np.identity(confusion.shape[1])

    # Get all TRUE negative values (TN). These are in the main diagonal of
    # (confusion * id_matrix), non-diagonal values are zeros:
    # We obtain a 1D vector result of the sum along the columns:
    TN = np.sum(confusion * id_matrix, axis=0)

    # Get ALL negatives (N), adding all values in confusion:
    # We obtain a 1D vector result of the sum along the columns:
    N = np.sum(confusion, axis=0)

    # Compute the SP value:
    return TN / N
