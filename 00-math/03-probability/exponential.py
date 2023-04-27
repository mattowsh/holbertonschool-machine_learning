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

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""

        if x < 0:
            return 0

        e = 2.7182818285
        lambtha = self.lambtha

        # PMF formula:
        return (lambtha * (e ** (-lambtha * x)))

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""

        if x < 0:
            return 0

        e = 2.7182818285
        lambtha = self.lambtha

        # CDF formula:
        return (1 - (e ** (-lambtha * x)))
