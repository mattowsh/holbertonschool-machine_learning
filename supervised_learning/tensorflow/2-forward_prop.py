#!/usr/bin/env python3
"""
Task 2. Forward Propagation
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

        x: placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer of
        the network
        activations: list containing the activation functions for each layer
        of the network

        Returns: the prediction of the network in tensor form
    """

    for i in range(len(layer_sizes)):
        placeholder = x if i == 0 else placeholder
        new_layer = create_layer(placeholder, layer_sizes[i], activations[i])

    return new_layer
