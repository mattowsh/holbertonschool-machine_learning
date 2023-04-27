#!/usr/bin/env python3
"""
Task 4. Moving Average
"""
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

        data: list of data to calculate the moving average of
        beta: weight used for the moving average
    """

    weights = 0
    wma = []

    for i in range(len(data)):
        weights = beta * weights + (1 - beta) * data[i]
        bias_correction = 1 - (beta ** (i + 1))
        wma.append(weights / bias_correction)
    return wma
