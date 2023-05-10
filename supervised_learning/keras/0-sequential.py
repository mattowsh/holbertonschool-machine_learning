#!/usr/bin/env python3
"""
Task 0. Sequential
"""
import tensorflow as tf
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function to build a neural network with the Keras library

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
    model = keras.Sequential()

    # Set the deep learning regularization technique:
    regularizer = keras.regularizers.L2(lambtha)

    # Define the model architecture contemplating all layers, nodes and
    # activation functions:
    for i in range(len(layers)):
        model.add(keras.layers.Dense(layers[i],
                                     input_shape=(nx, ),
                                     activation=activations[i],
                                     kernel_regularizer=regularizer))

        # Apply Dropout only in hidden layers, never in output layer:
        if i < (len(layers) - 1):
            model.add(keras.layers.Dropout(1 - keep_prob))

    return model
