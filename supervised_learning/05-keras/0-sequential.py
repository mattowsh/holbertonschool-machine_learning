#!/usr/bin/env python3
"""
Task 0. Sequential
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function to build a neural network with the Keras library without
    Input class

        nx: number of input features to the network
        layers: list containing the number of nodes in each layer of the
        network
        activations: list containing the activation functions used for each
        layer of the network
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns: the keras model
    """

    # Create the model structure:
    model = K.Sequential()

    # Set the deep learning regularization technique:
    regularizer = K.regularizers.l2(lambtha)

    # Define the model architecture contemplating all layers, nodes and
    # activation functions:
    for i in range(len(layers)):
        if i == 0:
            # First layer: receive nx input features:
            model.add(K.layers.Dense(layers[i],
                                     input_shape=(nx,),
                                     activation=activations[i],
                                     kernel_regularizer=regularizer))
        else:
            # Add all other hidden layers with Dropout and L2 techniques:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer))

    return model
