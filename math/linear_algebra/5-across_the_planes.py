#!/usr/bin/env python3
"""
Task 5. Across The Planes
"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise"""

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return

    result = []
    if len(mat1) and len(mat2):
        for i in range(len(mat1)):
            new_row = []
            for j in range(len(mat1[0])):
                new_row.append(mat1[i][j] + mat2[i][j])
            result.append(new_row)
    return result
