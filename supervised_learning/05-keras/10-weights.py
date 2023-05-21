#!/usr/bin/env python3
"""
Task 10. Save and Load Weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model's weights

        network: model whose weights should be saved
        filename: path of the file that the weights should be saved to
        save_format: format in which the weights should be saved
    """

    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights

        network: model to which the weights should be loaded
        filename: path of the file that the weights should be loaded from
    """

    network.load_weights(filename)
