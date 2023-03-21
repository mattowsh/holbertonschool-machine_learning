#!/usr/bin/env python3
"""
Task 2. Size Me Please
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""

    matrix_shape = []
    while type(matrix) != int:
        matrix_shape.append(len(matrix))
        matrix = matrix[0]

    return matrix_shape
