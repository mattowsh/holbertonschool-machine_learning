#!/usr/bin/env python3
"""
Task 15. Put it all together and what do you get?
"""
import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam
    optimization, mini-batch gradient descent, learning rate decay, and batch
    normalization
    
		Data_train: tuple, contain the training inputs and training labels
		Data_valid: tuple, contain the validation inputs and validation labels
		layers: list, contain the number of nodes in each layer of the network
		activation: list, contain the activation functions used for each layer
        of the network
		alpha: learning rate
		beta1: weight for the first moment of Adam Optimization
		beta2: weight for the second moment of Adam Optimization
		epsilon: small number used to avoid division by zero
		decay_rate: decay rate for inverse time decay of the learning rate
        (the corresponding decay step should be 1)
		batch_size: number of data points that should be in a mini-batch
		epochs: number of times the training should pass through the whole dataset
		save_path: path where the model should be saved to
        
    Notes
    > The input data does not need to be normalized as it has already been
    scaled to a range of [0, 1]
    > The training function should allow for a smaller final batch (a.k.a. use
    the entire training set)
    > The learning rate should remain the same within the an epoch (a.k.a. all
    mini-batches within an epoch should use the same learning rate)
    > Before each epoch,the training data will be shuffle
    """
