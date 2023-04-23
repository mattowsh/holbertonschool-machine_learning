#!/usr/bin/env python3
"""
Task 7. Evaluate
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network
    
        X: numpy.ndarray containing the input data to evaluate
        Y: numpy.ndarray containing the one-hot labels for X
        save_path: location to load the model from

        Returns: the network's prediction, accuracy, and loss, respectively
    """