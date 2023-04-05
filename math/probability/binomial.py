#!/usr/bin/env python3
"""
Binomial distribution
"""


def factorial(n):
    """Calculates the factorial of any n: n!"""
    n_factorial = 1
    for i in range(1, n + 1):
        n_factorial *= i
    return n_factorial


class Binomial():
    """Class that represent a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""

        if data is None:
            if n <= 0:
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

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""

        if k < 0:
            return 0
        if type(k) != int:
            k = int(k)

        n, p = self.n, self.p
        nk = (factorial(n)) / (factorial(k) * factorial(n - k))
        return (nk * (p ** k) * ((1 - p) ** (n - k)))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""

        if k < 0:
            return 0
        if type(k) != int:
            k = int(k)

        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

