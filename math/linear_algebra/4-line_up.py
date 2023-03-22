#!/usr/bin/env python3
matrix_shape = __import__('2-size_me_please').matrix_shape
"""
Task 4. Line Up
"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise"""

    if matrix_shape(arr1) != matrix_shape(arr2):
        return None

    result = []
    for i in range(len(arr1)):
        new_value = arr1[i] + arr2[i]
        result.append(new_value)
    return result
