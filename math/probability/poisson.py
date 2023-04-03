#!/usr/bin/env python3
"""
Task 0. Initialize Poisson
"""


class Poisson():
    """Class that represent a Poisson distribution"""

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
                # lambtha == mean of data:
                self.lambtha = float(sum(data) / len(data))
