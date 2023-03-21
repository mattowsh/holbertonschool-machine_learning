#!/usr/bin/env python3
"""
Task 2. Size Me Please
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""

    qty_layers = len(matrix)
    qty_rows = len(matrix[0])

    # Check if matrix is 2D or 3D:
    if isinstance(matrix[0][0], list):
        qty_cols = len(matrix[0][0])
        matrix_shape = [qty_layers, qty_rows, qty_cols]
    else:
        matrix_shape = [qty_layers, qty_rows]
    return matrix_shape
