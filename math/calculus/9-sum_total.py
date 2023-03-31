#!/usr/bin/env python3
"""
Task 9. Create your own sigma sum
"""


def summation_i_squared(n):
    """Function that calculate the squares of i (= 1), n times"""

    if type(n) != int or n < 1:
        return

    if n == 1:
        return n**2
    else:
        result = n**2
        n -= 1
        result += summation_i_squared(n)

    return result
