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

        if i == L:
            dz_current = grads["dz" + str(i)]
            dz_current = A_current - Y
        else:
            dz_prev = grads["dz" + str(i + 1)]
            dz_current = grads["dz" + str(i)]
            dz_current = np.matmul(W_current.transpose(), dz_prev) * (A_current * (1 - A_current))

        # Update weights and biases
        W_current = weights["W" + str(i)]
        b_current = weights["b" + str(i)]

        dW = (1 / m) * (np.matmul(dz_current, A_prev.transpose()) + (lambtha * W_current))
        db = (1 / m) * (np.sum(dz_current, axis=1, keepdims=True) + (lambtha * b_current))

        grads["dW" + str(i)], grads["db" + str(i)] = dW, db
        
        W_current -= (alpha * dW)
        b_current -= (alpha * db)
        
    return weights

