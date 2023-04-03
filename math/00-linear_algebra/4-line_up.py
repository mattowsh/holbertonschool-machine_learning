#!/usr/bin/env python3
"""
Task 4. Line Up
"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise"""

    if len(arr1) != len(arr2):
        return

    result = []
    if len(arr1):
        for i in range(len(arr1)):
            new_value = arr1[i] + arr2[i]
            result.append(new_value)
    return result
