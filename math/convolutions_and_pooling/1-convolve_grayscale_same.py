#!/usr/bin/env python3
"""
Task 1. Same Convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

        images: numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
        the convolution
            kh: height of the kernel
            kw: width of the kernel

    Note: if necessary, the image should be padded with 0's

    Returns: a numpy.ndarray containing the convolved images
    """

    k_shape = np.array(kernel.shape)

    # Add extra padding to the images:
    padding = np.ceil((k_shape - 1) / 2).astype(int)
    images_padded = np.pad(images, ((0, 0), (padding[0], padding[0]),
                           (padding[1], padding[1])), 'constant',
                           constant_values=0)

    # Create the result matrix that stores the convolved image:
    output_image = np.zeros(images.shape)

    # Apply the filter to the images == convolution:
    for i in range(output_image.shape[1]):
        for j in range(output_image.shape[2]):
            output_image[:, i, j] = np.sum(images_padded[:, i: i + k_shape[0],
                                           j: j + k_shape[1]] * kernel,
                                           axis=(1, 2))
    return (output_image)
