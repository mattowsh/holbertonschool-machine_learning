#!/usr/bin/env python3
"""
Task 0. Valid Convolution
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

        images: numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
        the convolution
            kh: height of the kernel
            kw: width of the kernel

    Returns: a numpy.ndarray containing the convolved images
    """

    # Get the dimensions of your images and kernel:
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculates the dimension of the output
    # (must be low than input dimensions):
    output_h = h - kh + 1
    output_w = w - kh + 1

    # Create the result matrix that stores the convolved image:
    output_image = np.zeros((m, output_h, output_w))

    # Apply the filter to the images == convolution:
    for i in range(output_h):
        for j in range(output_w):
            # Sum across the height and width dimensions:
            output_image[:, i, j] = np.sum((images[:, i: i + kh, j: j + kw]
                                            * kernel), axis=(1, 2))

    return output_image
