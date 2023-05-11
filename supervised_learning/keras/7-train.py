#!/usr/bin/env python3
"""
7. Learning Rate Decay
(based on 6-train.py)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
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

        - learning_rate_decay: boolean, indicates whether learning rate decay
        should be used
            Note: should only be performed if validation_data exists
            Note: the decay should be performed using inverse time decay
            Note: should decay in a stepwise fashion after each epoch
            Note: each time the learning rate updates, Keras should print
            a message

        - alpha: initial learning rate
        - decay_rate: decay rate

        - verbose: boolean that determines if output should be printed during
        training
        - shuffle: boolean that determines whether to shuffle the batches every
        epoch

        Note: Normally, it is a good idea to shuffle, but for reproducibility,
        we have chosen to set the default to False

        Return: the History object generated after training the model
    """

    def learning_rate(epoch):
        """
        Function that calculate the learning rate

        Note: is a mandatory parameter to the LearningRateScheduler
        Keras callback
        """

        return alpha / (1 + decay_rate * epoch)

    if validation_data:
        custom_callback = []

        if early_stopping:
            custom_callback.append(
                K.callbacks.EarlyStopping(monitor="loss", patience=patience))

        if learning_rate_decay:
            # Verbose parameter: update Keras messages:
            custom_callback.append(
                K.callbacks.LearningRateScheduler(learning_rate, verbose=1))
    else:
        custom_callback = None

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=custom_callback,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
