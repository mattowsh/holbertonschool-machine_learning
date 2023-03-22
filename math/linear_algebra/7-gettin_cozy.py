#!/usr/bin/env python3
"""
Task 7. Gettin' Cozy
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.
    mat1 and mat2 are 2D matrices containing ints/floats
    """

    # Add the new values as a new row => axis == 0
    if not axis:
        return mat1 + mat2
    
    # Add the new values as a new columns => axis == 1
    if axis == 1 and len(mat1) == len(mat2):
        result = []
        result = [(mat1[i] + mat2[i]) for i in range(len(mat1))]
    else:
        return
