#!/usr/bin/env python3
"""
Task 3. Specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

        confusion: confusion numpy.ndarray (classes, classes)
            classes: number of classes

    Return: a numpy.ndarray (classes,) containing the specificity of each class
    """

    # specifity (SP) == true negative rate (TNR)
    # number of correct negative predictions (TN) divided by the total number
    # of negatives (N) => TN / N

    id_matrix = np.identity(confusion.shape[1])
    total = np.sum(confusion)

    # Calculate true positives:
    TP = np.sum(confusion * id_matrix, axis=0)

    # Calculate false positives:
    FP = np.sum(confusion - (confusion * id_matrix), axis=0)

    # Calculate true negatives:
    TN = total - (FP + np.sum(confusion, axis=1))

    # Calculate false negstives:
    FN = np.sum(confusion, axis=0) - TP

    # Calculate all N = TN + FP
    return TN / (TN + FP)
    return TN / (TN + FP)
