#!/usr/bin/env python3
"""
Write a function that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):

    t_matrix = [[matrix[i][j] for i in range(len(matrix))]
                for j in range(len(matrix[0]))]
    return (t_matrix)
