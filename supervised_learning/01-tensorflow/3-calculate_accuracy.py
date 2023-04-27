#!/usr/bin/env python3
"""
Task 3. Accuracy
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction

        y: placeholder for the labels of the input data
        y_pred: tensor containing the network's prediction

        Return: a tensor containing the decimal accuracy of the prediction
    """

    # Convert the predicted probabilities into predicted class labels:
    predicted_classes = tf.argmax(y_pred, axis=1)

    # Convert the real labes into class labels:
    true_classes = tf.argmax(y, axis=1)

    # Comparte the predicted classes vs. the real classes:
    classes_comparison = tf.equal(predicted_classes, true_classes)

    # Cast the boolean values to float32 dtype:
    cast_values = tf.cast(classes_comparison, tf.float32)

    # Calculate the mean of the casted values:
    accuracy = tf.reduce_mean(cast_values)

    return accuracy
