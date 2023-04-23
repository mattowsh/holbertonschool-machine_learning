#!/usr/bin/env python3
"""
Task 6. Train_Op
"""
import tensorflow as tf
# Import previos functions:
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier

        X_train: numpy.ndarray containing the training input data
        Y_train: numpy.ndarray containing the training labels
        X_valid: numpy.ndarray containing the validation input data
        Y_valid: numpy.ndarray containing the validation labels
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        alpha: learning rate
        iterations: number of iterations to train over
        save_path: file to save the model
    """

    # Initialize TensorFlow's random seed to a fixed value:
    tf.set_random_seed(0)

    # Clear the previous TensorFlow graph to reset it:
    tf.reset_default.graph()

    # Set the input and output placeholders:
    X, Y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Create the neural network:
    Y_pred = forward_prop(X, layer_sizes, activations)

    # Calculate loss and accuracy of the neural network:
    loss = calculate_loss(Y, Y_pred)
    accuracy = calculate_accuracy(Y, Y_pred)

    # Create the training operation:
    train_op = create_train_op(loss, alpha)

    # New TensorFlow session + initialization of variables:
    init = tf.global_variables_initializer()
    with tf.Session() as s:
        s.run(init)

        # Simplify the access of essential variables in our graph's
        # collection:
        tf.add_to_collection('X', X)
        tf.add_to_collection('Y', Y)
        tf.add_to_collection('Y_pred', Y_pred)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('accuracy', accuracy)
        tf.add_to_collection('train_op', train_op)

        # Train the model:
        for i in range(iterations + 1):
            cost_train, acc_train = s.run(
                [loss, accuracy], feed_dict={X: X_train, Y: Y_train})
            cost_valid, acc_valid = s.run(
                [loss, accuracy], feed_dict={X: X_valid, Y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(acc_valid))

            if i < iterations:
                s.run(train_op, feed_dict={X: X_train, Y: Y_train})

        # Save the trained model using a TensorFlow Saver object:
        save_model = tf.train.Saver()
        save_path = save_model.save(s, save_path)

    return save_path
