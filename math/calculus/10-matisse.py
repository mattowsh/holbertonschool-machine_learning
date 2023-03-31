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
    i = 1
    try:
        while i < len(poly):
            result.append(i * poly[i])
    except Exception:
        return
    
    # for i in poly:
    #     if type(i) != int:
    #         return

    #     if poly.index(i) == 0:
    #         continue
    #     result.append(i * poly.index(i))

    # Final returns:
    if len(result) != 0:
        return result
    else:
        return
