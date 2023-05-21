#!/usr/bin/env python3
"""
Task 9. Save and Load Model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model

    network: model to save
    filename: path of the file that the model should be saved to

    Returns: None
    """

    network.save(filename)


def load_model(filename):
    """
    Loads an entire model

    filename: path of the file that the model should be loaded from

    Returns: the loaded model
    """

    return K.models.load_model(filename)
