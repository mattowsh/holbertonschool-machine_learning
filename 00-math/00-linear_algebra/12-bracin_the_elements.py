#!/usr/bin/env python3
"""
Task 12. Bracing The Elements
"""


def np_elementwise(mat1, mat2):
    """
    > Performs element-wise addition, subtraction, multiplication,
    and division
    > mat1 and mat2 can be interpreted as numpy.ndarrays, and never empty
    """

    add, sub = mat1 + mat2, mat1 - mat2
    mul, div = mat1 * mat2, mat1 / mat2
    return add, sub, mul, div
