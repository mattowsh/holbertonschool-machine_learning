#!/usr/bin/env python3
"""
Task 7. Evaluate
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network

        X: numpy.ndarray containing the input data to evaluate
        Y: numpy.ndarray containing the one-hot labels for X
        save_path: location to load the model from

        Returns: the network's prediction, accuracy, and loss, respectively
    """

    # Load the training model:
    with tf.Session() as sess:
        # Import the metagraph and restore the w values:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        # Get the data restored:
        X = tf.get_collection("X")[0]
        Y = tf.get_collection("Y")[0]
        Y_pred = tf.get_collection("Y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        # Run the evaluation:
        Y_eval = sess.run(Y_pred, feed_dict={X: X, Y: Y})
        accuracy_eval = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss_eval = sess.run(loss, feed_dict={x: X, y: Y})

        # Final return:
        return Y_eval, accuracy_eval, loss_eval
