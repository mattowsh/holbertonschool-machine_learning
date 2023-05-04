#!/usr/bin/env python3
"""
Task 11. Learning Rate Decay
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy

        alpha: the original learning rate
        decay_rate: weight used to determine the rate at which alpha
        will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further

        Note: the learning rate decay should occur in a stepwise fashion
    """

    # Use np.floor() to round down the result to the nearest integer.
    # This is necessary because the decay rate should only be applied after a
    # certain number of steps == decay_step:
    steps = np.floor(global_step / decay_step)

    new_lr = alpha / (1 + decay_rate * steps)
    return new_lr
