#!/usr/bin/env python3
"""
Task 9. Create your own sigma sum
"""


def summation_i_squared(n):
    """Function that calculate the squares of i (= 1), n times"""

    if type(n) != int:
        return

    if n == 0:
        return int(0)
    else:
        result = n**2
        n -= 1
        result += summation_i_squared(n)

    return result
