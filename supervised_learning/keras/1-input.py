#!/usr/bin/env python3
"""
Task 1. Input
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function to build a neural network with the Keras library without
    Sequential class

        nx: number of input features to the network
        layers: list containing the number of nodes in each layer of the
        network
        activations: list containing the activation functions used for each
        layer of the network
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns: the keras model
    """
    
	