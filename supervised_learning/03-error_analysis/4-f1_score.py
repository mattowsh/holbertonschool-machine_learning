#!/usr/bin/env python3
"""
Task 4. F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix

        confusion: confusion numpy.ndarray (classes, classes)
            classes: number of classes

    Return: a numpy.ndarray (classes,) containing the F1 score of each class
    """

    # f score: the harmonic mean of the precision and recall

    sn = sensitivity(confusion)
    prec = precision(confusion)
    fscore = 2 * ((prec * sn) / (prec + sn))

    return fscore
