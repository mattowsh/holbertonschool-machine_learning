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

    # precision (PREC) == positive predictive value (PPV)
	# the number of correct positive predictions (TP) divided by the total
	# number of positive predictions (P = TP + FP)

    id_matrix = np.identity(confusion.shape[1])

    # Get all TRUE positive values (TP). These are in the main diagonal of
    # (confusion * id_matrix), non-diagonal values are zeros:
    # We obtain a 1D vector result of the sum along the columns:
    TP = np.sum(confusion * id_matrix, axis=0)

    # Count the number of true positives for each PREDICTED class
    # We obtain a 1D vector result of the sum along the columns:
    TPsumFP = np.sum(confusion, axis=0)

    # Compute the SP value:
    return TP / TPsumFP
