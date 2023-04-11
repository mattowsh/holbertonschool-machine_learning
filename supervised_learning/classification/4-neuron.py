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

    def forward_prop(self, X):
        """
        Calculates the fôrward propagation of the neuron.

            X: a numpy.ndarray with shape (nx, m) that contains the input data
            nx: number of input features to the neuron
            m: number of examples
        """

        matrix_result = np.matmul(self.__W, X) + self.__b
        # Activation function: sigmoid
        self.__A = 1 / (1 + np.exp(-matrix_result))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

            Y: numpy.ndarray with shape (1, m), contains the correct labels
            for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
            of the neuron for each example
        """

        # Firstly, calculate the cross-entropy == loss function:
        loss = -((Y * np.log(A)) + (1 - Y) * (np.log(1.0000001 - A)))

        # Secondly, calculate the cost function == average of each loss result:
        return loss.mean()

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

            X: numpy.ndarray with shape (nx, m), contains the input data
            Y: numpy.ndarray with shape (1, m), contains the correct labels
            for the input data
        """

        predicted_values = self.forward_prop(X)
        cost = self.cost(Y, predicted_values)

        for i in range(len(predicted_values)):
            if predicted_values[i] >= 0.5:
                predicted_values[i] == 1
            else:
                predicted_values[i] == 0

        return predicted_values, cost
