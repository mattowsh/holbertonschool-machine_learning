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

    # Set the deep learning regularization technique:
    regularizer = K.regularizers.l2(lambtha)

    # Define the input layer:
    input_layer = K.Input(shape=(nx,))
    other_layers = input_layer

    # Define the model architecture:
    for i in range(len(layers)):
        if i != 0:
            # Define hidden and output layers. Input layer (layers[0]) does not
            # contain Dropout:
            other_layers = K.layers.Dropout(1 - keep_prob)(other_layers)

        other_layers = K.layers.Dense(layers[i],
                                      activation=activations[i],
                                      kernel_regularizer=regularizer
                                      )(other_layers)

    # Create the model:
    model = K.Model(inputs=input_layer, outputs=other_layers)
    return model
