#!/usr/bin/env python3
"""
Task 12. Learning Rate Decay Upgraded
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse
    time decay

        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha
        will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further

    Note: the learning rate decay should occur in a stepwise fashion
    """

    # Define the operation:
    # staircase parameter: whether to apply decay in a discrete staircase,
    # as opposed to continuous, fashion
    lr_decay_op = tf.train.inverse_time_decay(
        learning_date=alpha,
        global_step=global_step,
        decay_step=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    # Final return:
    return lr_decay_op
