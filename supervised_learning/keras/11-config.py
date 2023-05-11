#!/usr/bin/env python3
"""
Task 11. Save and Load Configuration
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format

        network: model whose configuration should be saved
        filename: path of the file that the configuration should be saved to
    """

    # Turn into JSON file:
    json_model = network.to_json()

    with open(filename, "w") as file:
        file.write(json_model)


def load_config(filename):
    """
    Loads a model with a specific configuration

        filename:  file containing the model's configuration in JSON format

    Returns: the loaded model
    """

    # Read the JSON file:
    with open(filename, "r") as file:
        json_model = file.read()

    # Create the Keras model:
    loaded_model = K.model_from_json(json_model)

    return loaded_model
