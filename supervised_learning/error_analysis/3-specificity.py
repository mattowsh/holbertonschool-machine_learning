#!/usr/bin/env python3
"""
Task 
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    
    	confusion: confusion numpy.ndarray (classes, classes)
            classes: number of classes
            
    Return: a numpy.ndarray (classes,) containing the specificity of each class
    """
    
	# specifity (SP) == true negative rate (TNR)
    # number of correct negative predictions (TN) divided by the total number
    # of negatives (N) => TN / N

	