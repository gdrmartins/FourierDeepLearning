#pylint: disable=C0103

"""

Mark Fourier Module

This module provides methods that are used to compute the fast fourier
transform of any given value using the Cooley-Tuckey algorithm.

Given that this module was implemented for academic purposes, this fast
fourier transform implementation supports power of two lengths of up to
16384.

Example:
    Here is an example that will be enough for 99% of users:

        >>> import mark_fourier as fourier
        >>> fourier.fft([0, 1, 2, 3])
        >>> fourier.ifft([0, 1, 2, 3])

Attributes:
    POW2 : list
        This attribute holds all currently supported lengths of
        data for the fft and ifft calculations.

Requriements:
    * Numpy ~1.13.3

"""

import numpy as np


POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def bisect(array, length, value):
    """
    This method is used to find the index of a sorted array where
    a given value should be inserted into to keep the order.

    Properties:
        * array : list
            A slice of the POW2 array where the algorithm must look
            for the target index.
        * length : int
            The length of the array (POW2 slice).
        * value : int
            The value whose insert index needs to be found.
    """
    half = int(length / 2)
    if length < 3:
        return array[1]
    elif value > array[half]:
        return bisect(array[half:], half, value)
    elif value < array[half]:
        return bisect(array[:half], half, value)
    return value


def next_pow_two(target):
    """
    This method is used to find the next power of two number of any
    given value.

    Properties:
        * target : int
            The value whose next power of two needs to be found.
    """
    return bisect(POW2, len(POW2), target)


def dft(array):
    """
    This method is used to compute the dft of a given 1D array.

    Properties:
        * array : list
            1D array containing all the data that must be transformed
            using the discrete fourier transform.
    """
    # Define main variables of the dft transform
    x = np.asarray(array, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Generate a matrix with all exponential values
    M = np.exp(-2j * np.pi * k * n / N)
    # Calculate the sum of the product of x*M
    return np.dot(M, x)


def fft(array):
    """
    This method is used to compute the fft of a given array using
    the Cooley-Tuckey algorithm. Supports lengths of up to 16384.

    Properties:
        * array : list
            1D array containing all the data that must be transformed
            using the Cooley-Tuckey fast fourier transform.
    """
    # Make sure the length of the array is power of two
    current_length = len(array)
    target_length = next_pow_two(current_length)
    delta = target_length - current_length
    for _ in range(delta):
        array.insert(0, 0)
    # Compute the transform
    return transform(array)


def transform(array):
    """
    This is the recursive part of the fft method, the part that
    actually computes the transform

    Properties:
        * array : list
            1D array containing all the data that must be transformed
            using the Cooley-Tuckey fast fourier transform.
    """
    # Define main variables of the fft transform
    x = np.asarray(array, dtype=float)
    N = x.shape[0]
    # Compute the dft when the length is small enough
    if N <= 2:
        return dft(array)
    # Devide the problem in even and odd indexes and recurse
    even = transform(x[::2])
    odd = transform(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + factor[:int(N / 2)] * odd, even + factor[int(N / 2):] * odd])
