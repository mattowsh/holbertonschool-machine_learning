#!/usr/bin/env python3
"""
Tasks: Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork():
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor class

            nx: number of input features
            layers: number of nodes in each layer of the network"""

        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Check if all elements are positive integers:
        # Status true if any value in layers is <= 0:
        status = any(i <= 0 for i in layers)
        if status:
            raise TypeError("layers must be a list of positive integers")

        # Number of layers in the neural network:
        self.L = len(layers)
        # A dictionary to hold all intermediary values of the network:
        self.cache = {}
        # A dictionary to hold all weights and biased of the network:
        self.weights = {}

        for i in range(1, len(layers)):
            # He et al. initialization:
            he_init = np.sqrt(2 / layers[i - 1])

            # Get weights and bias randomly:
            Wi = np.random.randn(layers[i], layers[i - 1]) * he_init
            bi = np.zeros((layers[i], 1))

            # Set the values in weights dict:
            self.weights["W{}".format(i)] = Wi
            self.weights["b{}".format(i)] = bi
