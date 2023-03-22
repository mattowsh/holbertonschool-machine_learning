#!/usr/bin/env python3
"""
Task 8. Ridin' Bareback
"""


def mat_mul(mat1, mat2):
    """
    > Performs matrix multiplication
    > mat1 and mat2 are 2D matrices containing ints/floats
    """

    m1_rows, m1_cols = len(mat1), len(mat1[0])
    m2_rows, m2_cols = len(mat2), len(mat2[0])

    if m1_cols != m2_rows:
        return

    # Create a full-zeros matrix to save our results:
    result = [[0 for x in range(m2_cols)] for y in range(m1_rows)]

    # Calculate each result:
    for i in range(m1_rows):
        for j in range(m2_cols):
            for k in range(m1_cols):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
