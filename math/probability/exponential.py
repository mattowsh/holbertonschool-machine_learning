#!/usr/bin/env python3
"""
Exponential Distribution
"""


class Exponential():
    """Class that represent an Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""

        if data is None:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError("lambtha must be a positive value")

        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # lambda == inverse of the mean
                self.lambtha = float(1 / (sum(data) / len(data)))
