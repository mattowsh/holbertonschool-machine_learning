#!/usr/bin/env python3
"""
Tasks: Neural Network
"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing
    binary classification"""

    def __init__(self, nx, nodes):
        """Constructor class

            nx: number of input features
            nodes: number of nodes in the hidden layer"""

        # Parameters validation:
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # 1st neuron attributes:
        self.W1 = np.random.normal(size=(1, nx))
        self.b1 = 0
        self.A1 = 0
        # 2nd neuron attributes:
        self.W2 = np.random.normal(size=(1, nx))
        self.b2 = 0
        self.A2 = 0
