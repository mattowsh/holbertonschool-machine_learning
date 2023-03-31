#!/usr/bin/env python3
"""
Task 17. Integral of a polynomial
"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""

    if type(poly) != list or type(C) != int or len(poly) == 0:
        return

    index, result = 0, [C]
    if poly == [0]:
        return result

    for i in range(len(poly)):
        value = poly[i] / (index + 1)

        if int(value) == value:
            result.append(int(value))
        else:
            result.append(value)
        index += 1

    return result
