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

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""

        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0

        # Define constans in function to PMF equation:
        e = 2.7182818285
        lambtha = self.lambtha
        i, k_factorial = 1, 1

        # Calculates k!:
        while (i <= k):
            k_factorial *= i
            i += 1

        # Calculates PMF value for k:
        pmf_value = (lambtha ** k) * (e ** (-lambtha)) / k_factorial
        return pmf_value

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""

        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0

        # Define constans in function to CDF equation:
        e = 2.7182818285
        lambtha = self.lambtha
        subproduct, j = 0, 0

        # Calculates CDF value for k:
        while (j <= k):
            i, i_factorial = 1, 1
            while (i <= j):
                i_factorial *= i
                i += 1

            subproduct += ((lambtha ** j) / i_factorial)
            j += 1

        cdf_result = (e ** (-1 * lambtha)) * subproduct
        return cdf_result
