#!/usr/bin/env python3
"""
Task 10. Derivate of a polyomial
"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""

    if type(poly) != list:
        return

    # Derivate of a int == 0, always cte.
    if len(poly) == 1:
        return [0]

    result = []
    for i in poly:
        if type(i) != int:
            return

        if poly.index(i) == 0:
            continue
        sub_result = i * poly.index(i)
        result.append(sub_result)

    # Final returns:
    result if (len(result) != 0) else None
