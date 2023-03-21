#!/usr/bin/env python3
"""
Task 3. Flip Me Over
"""


def matrix_transpose(matrix):
    """Returns the transpose of a matrix"""
    # Fst alternative:
    # t_matrix = [[matrix[i][j] for i in range(len(matrix))]
    #             for j in range(len(matrix[0]))]

    # Snd alternative:
    t_matrix = []
    rows = len(matrix)
    columns = len(matrix[0])

    for i in range(rows):
        new_row = []
        for j in range(columns):
            new_row.append(matrix[j][i])
        t_matrix.append(new_row)

    return (t_matrix)
