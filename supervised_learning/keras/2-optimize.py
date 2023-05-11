#!/usr/bin/env python3
"""
Task 2. Optimize
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical crossentropy
    loss and accuracy metrics

        network: model to optimize
        alpha: learning rate
        beta1: first Adam optimization parameter
        beta2: second Adam optimization parameter

    Returns: None
    """

    # Define the Adam optimizer:
    adam_op = K.optimizers.Adam(lr=alpha,
                                beta_1=beta1,
                                beta_2=beta2)

    # Compile the model:
    network.compile(loss="categorical_crossentropy",
                    optimizer=adam_op,
                    metrics=["accuracy"])
