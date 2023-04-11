#!/usr/bin/env python3
"""
Task 0. Neuron
"""
import numpy as np


class Neuron():
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Class constructor"""

        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.nx = nx

        # The weights vector for the neuron
        self.W = np.random.normal(size=(1, nx))
        # The bias for a neuron
        self.b = 0
        # The activated output of the neuron (prediction)
        self.A = 0
