#!/usr/bin/env python3
"""
Task 1. Gradient Descent with L2 Regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization

        Y: one-hot numpy.ndarray, contains the correct labels for the data
            classes: number of classes
            m: number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network

    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    """

    # Calculate the qty of training examples:
    m = Y.shape[1]

    # Get the gradient descent values:
    grads = {}

    # Backpropagation
    for i in range(L, 0, -1):
        A_current, A_prev = cache["A" + str(i)], cache["A" + str(i - 1)]
        # W_current, W_prev = weights["W" + str(i)], weights["W" + str(i - 1)]
        dZ = grads["dz" + str(i)]

        if i == L:
            grads[dZ] = cache[A_current] - Y
        else:
            dZ = np.dot(W_prev.T, dZ) * (1 - np.power(A, 2))

        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W_current
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        grads["dW" + str(i)], grads["db" + str(i)] = dW, db

    # Update weights and biases
    for i in range(1, L + 1):
        W_current = weights["W" + str(i)]
        b_current = weights["b" + str(i)]
        
        W_current -= (alpha * dW)
        b_current -= (alpha * db)
        
    return weights

