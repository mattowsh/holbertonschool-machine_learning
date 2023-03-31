#!/usr/bin/env python3
"""
Task 9. Create your own sigma sum
"""


def summation_i_squared(n):
    """Function that calculate the squares of i (= 1), n times"""

    if type(n) != int or n < 1:
        return

    return int(n * (n + 1) * (2 * n + 1)/6)

    # Recursion work but max calls in Python: 1000, stack overflow :^)
    # if n == 1:
    #     return n**2
    # else:
    #     result = n**2
    #     n -= 1
    #     result += summation_i_squared(n)

    # return result
