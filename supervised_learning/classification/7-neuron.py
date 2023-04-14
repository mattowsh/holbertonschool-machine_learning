#!/usr/bin/env python3
"""
Tasks: Neuron Class
    W: weights vector for the neuron
    b: bias for the neuron
    A: activated output of the neuron (prediction)
"""
import numpy as np
import matplotlib.pyplot as plt


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

        predictions = self.forward_prop(X)
        cost = self.cost(Y, predictions)

        return np.round(predictions).astype(int), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

            X: numpy.ndarray with shape (nx, m), contains the input data
            Y: numpy.ndarray with shape (1, m), contains the correct labels
            for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
            of the neuron for each example
            alpha: learning rate
        """

        # We use the backward algorithm:
        error = A - Y
        new_w = np.matmul(error, X.T) / X.shape[1]
        new_b = np.sum(error) / X.shape[1]

        # Update the parameters:
        self.__W = self.__W - alpha * new_w
        self.__b = self.__b - alpha * new_b

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron

            iterations: number of iterations to train over
            alpha: learning rate
            verbose: boolean, defines whether or not to print information about
            the training
            graph: boolean, defines whether or not to graph information about
            the training once the training has completed
        """

        # Parameter validations:
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        steps = []
        costs = []
        for i in range(iterations + 1):
            # A is a np.ndarray that contains all predicted values
            # Here will update the A attribute:
            A = self.forward_prop(X)

            # Update W and b in function to the product of the model train:
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

            # Create the necessary arrays to create the chart:
            # Calculate costs only just each "step" steps, this avoid us
            # use plt.xticks in the chart construction:
            if i % step == 0:
                cost = self.cost(Y, A)
                steps.append(i)
                costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, cost))

        if graph:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")

            plt.plot(steps, costs)
            plt.show()

        # Final return: evaluation of the model performance:
        return self.evaluate(X, Y)
