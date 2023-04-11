#!/usr/bin/env python3
"""
Tasks: Neuron Class
"""
import numpy as np


class Neuron():
    """Defines a single neuron perfôrming binary classification"""

    def __init__(self, nx):
        """Class constructor"""

        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.nx = nx

            # Weights vector
            self.W = np.random.normal(size=(1, nx))
            # Bias
            self.b = 0
            # The activated output of the neuron (prediction)
            self.A = 0
