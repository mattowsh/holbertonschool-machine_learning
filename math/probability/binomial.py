#!/usr/bin/env python3
"""
Binomial distribution
"""


class Binomial():
    """Class that represent a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""

        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = sum(data) / len(data)
                result = 0
                for i in range(len(data)):
                    result += (data[i] - mean) ** 2
                variance = result / len(data)

                self.n = round(mean / (-(variance/mean) + 1))
                # p == qty successes / qty of trials (n)
                self.p = mean / self.n
