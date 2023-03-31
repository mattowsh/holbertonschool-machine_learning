#!/usr/bin/env python3
"""
Task 10. Derivate of a polyomial
"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""

    if type(poly) != list and type(poly) != tuple:
        return

    # Derivate of a int == 0, always cte.
    if len(poly) == 1:
        return [0]

    result = []
    for i in range(len(poly)):
        if type(poly[i]) != int:
            return

        if i == 0:
            continue
        result.append(i * poly[i])

    ternary_result = result if len(result) != 0 else None
    return ternary_result
