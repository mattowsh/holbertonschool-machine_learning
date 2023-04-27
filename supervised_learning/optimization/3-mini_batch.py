#!/usr/bin/env python3
"""
Task 3. Mini-Batch
"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
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
        epochs: number of times the training should pass through the whole dataset
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
        
        # Get the necessary tensors and operations from the collection
        # restored:
        # placeholder for the input data:
        x = tf.get_collection("x")[0]
        # placeholder for the labels:
        y = tf.get_collection("y")[0]
        # op to calculate the accuracy of the model:
        accuracy = tf.get_collection("accuracy")[0]
        # op to calculate the cost of the model:
        loss = tf.get_collection("loss")[0]
        # op to perform one pass of gradient descent on the model:
        # train_op = tf.get_collection("")[0]
        
        for e in tf.get_collection(c):
        	print("\t{}".format(e.name))
        
def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 5000

    np.random.seed(0)
    save_path = train_mini_batch(X_train, Y_train_oh, X_valid, Y_valid_oh,
                                 epochs=10, load_path='./graph.ckpt',
                                 save_path='./model.ckpt')

    
