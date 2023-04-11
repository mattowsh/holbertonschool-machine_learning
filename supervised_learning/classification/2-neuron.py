#!/usr/bin/env python3
"""
Tasks: Neuron Class
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
            self.__W = np.random.normal(size=(1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """Private W attribute getter function"""
        return self.__W

    @property
    def b(self):
        """Private b attribute getter function"""
        return self.__b

    @property
    def A(self):
        """Private A attribute getter function"""
        return self.__A
