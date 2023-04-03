#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal():
    """Class that represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor"""

        if data is None:
            if stddev > 0:
                self.stddev = float(stddev)
                self.mean = float(mean)
            else:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # We have data, so, we can estimate mean and stddev:
                self.mean = float(sum(data) / len(data))

                stddev_result = 0
                for i in range(len(data)):
                    stddev_result += ((data[i] - self.mean) ** 2)
                # Remember: std desv**2 == variance:
                self.stddev = (stddev_result / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (self.mean + (z * self.stddev))
