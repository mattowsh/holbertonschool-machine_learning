#!/usr/bin/env python3
"""
Task 2. Convolution with Padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

        images: numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
        the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding: tuple of (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    """

    k_shape = np.array(kernel.shape)
    p_shape = np.array(padding)

    # Add extra padding to the images based in the custom padding:
    images_padded = np.pad(images, ((0, 0), (p_shape[0], p_shape[0]),
                           (p_shape[1], p_shape[1])), 'constant',
                           constant_values=0)

    # Calculates output dimensions:
    output_h = images.shape[1] + (2 * p_shape[0]) - k_shape[0] + 1
    output_w = images.shape[2] + (2 * p_shape[1]) - k_shape[1] + 1

    # Create the result matrix that stores the convolved image:
    output_image = np.zeros((images.shape[0], output_h, output_w))

    # Apply the filter to the images == convolution:
    for i in range(output_image.shape[1]):
        for j in range(output_image.shape[2]):
            output_image[:, i, j] = np.sum(images_padded[:, i: i + k_shape[0],
                                           j: j + k_shape[1]] * kernel,
                                           axis=(1, 2))
    return (output_image)
