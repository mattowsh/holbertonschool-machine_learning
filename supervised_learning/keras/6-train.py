#!/usr/bin/env python3
"""
Task 6. Early Stopping
(based on 5-train.py)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

        - network: model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        - labels: one-hot numpy.ndarray of shape (m, classes) containing the
        labels of data
        - batch_size: size of the batch used for mini-batch gradient descent
        - epochs: number of passes through data for mini-batch gradient descent

        - validation_data: data to validate the model with

        - early_stopping: boolean, indicates whether early stopping should be
        used
            Note: should only be performed if validation_data exists
            Note: should be based on validation loss
        - patience: patience used for early stopping

        - verbose: boolean that determines if output should be printed during
        training
        - shuffle: boolean that determines whether to shuffle the batches every
        epoch

        Note: Normally, it is a good idea to shuffle, but for reproducibility,
        we have chosen to set the default to False

        Return: the History object generated after training the model
    """

    if validation_data and early_stopping:
        custom_callback = []
        custom_callback.append(
            K.callbacks.EarlyStopping(monitor="loss", patience=patience))
    else:
        custom_callback = None


    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callback=custom_callback,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
