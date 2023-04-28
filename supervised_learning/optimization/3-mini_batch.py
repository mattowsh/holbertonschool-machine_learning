#!/usr/bin/env python3
"""
Task 3. Mini-Batch
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent

        X_train: numpy.ndarray (m, 784) containing the training data
            m: number of data points
            784: number of input features
        Y_train: one-hot numpy.ndarray (m, 10) containing the training labels
        10: number of classes the model should classify
        X_valid: numpy.ndarray (m, 784) containing the validation data
        Y_valid: one-hot numpy.ndarray (m, 10) containing the validation labels
        batch_size: number of data points in a batch
        epochs: number of times the training should pass through the whole
        dataset
        load_path: path from which to load the model
        save_path: path to where the model should be saved after training

        Note: the training function should allow for a smaller final batch
        (a.k.a. use the entire training set) :)
    """

    # Load the training model:
    with tf.Session() as sess:
        # Import the metagraph and restore the w values:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        # Get the necessary tensors and operations:
        # placeholder for the input data:
        x = tf.get_collection("x")[0]
        # placeholder for the labels:
        y = tf.get_collection("y")[0]
        # op to calculate the accuracy of the model:
        accuracy = tf.get_collection("accuracy")[0]
        # op to calculate the cost of the model:
        loss = tf.get_collection("loss")[0]
        # op to perform one pass of gradient descent on the model:
        train_op = tf.get_collection("train_op")[0]

        # Calculate the number of batches:
        qty_datapoint = X_train.shape[0]

        if qty_datapoint % batch_size == 0:
            qty_batches = qty_datapoint // batch_size
        else:
            qty_batches = qty_datapoint // batch_size + 1

        # Loop over epochs:
        for i in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            # Print information:
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if i < epochs:
                # Shuffle data:
                X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)

                # Loop over the batches:
                for j in range(qty_batches):
                    # Train the model using mini-batches:
                    start = j * batch_size
                    end = (j + 1) * batch_size
                    if end > qty_datapoint:
                        end = qty_datapoint

                    # Get X_batch and Y_batch from data:
                    X_mini_batch = X_shuffle[start:end]
                    Y_mini_batch = Y_shuffle[start:end]

                    # Set the next mini-batch:
                    next_mbatch = {x: X_mini_batch, y: Y_mini_batch}

                    # Train the model using the mini-batches:
                    sess.run(train_op, feed_dict=next_mbatch)

                    # Print information about mini-batches results:
                    if j != 0 and (j + 1) % 100 == 0:
                        mbatch_cost = sess.run(loss, feed_dict=next_mbatch)
                        mbatch_acc = sess.run(accuracy, feed_dict=next_mbatch)

                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(mbatch_cost))
                        print("\t\tAccuracy: {}".format(mbatch_acc))

    # Save the trained model:
    return saver.save(sess, save_path)
