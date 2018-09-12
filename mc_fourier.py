"""

MC Fourier Module

This module provides methods that are used to compute the fast fourier
transform of any given value using the Cooley-Tuckey algorithm.

Given that this module was implemented for academic purposes, this fast
fourier transform implementation supports only power of two lengths.

Example:
    Here is an example that will be enough for 99% of users:

        >>> import mc_fourier as fourier
        >>> fourier.fft([0, 1, 2, 3])
        >>> fourier.ifft([0, 1, 2, 3])

Requriements:
    * Numpy ~1.13.3

"""

import numpy as np


def dft(array):
    """
    This method is used to compute the dft of a given 1D array.

    Properties:
        * array : list
            1D array containing all the data that must be transformed
            using the discrete fourier transform.
    """
    # Define main variables of the dft transform
    x = np.asarray(array, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Generate a matrix with all exponential values
    M = np.exp(-2j * np.pi * k * n / N)
    # Return the multiplication of the two matrices
    return np.dot(M, x)


def fft(array):
    """
    This method is used to compute the fft of a given array using the
    Cooley-Tuckey algorithm (radix-2 DIT). Supports lengths of up to
    1048576 or 1024x1024.

    Properties:
        * array : list
            1D array containing all the data that must be transformed
            using the Cooley-Tuckey fast fourier transform.
    """
    # Get input length
    length = len(array)
    # If input is not power of two, compute DFT
    if np.log2(length) % 1 > 0:
        return dft(array)
    # Define main variables of the FFT transform
    x = np.asarray(array, dtype=complex)
    # Determine the chunk size (32 is the recommendation)
    chunk_size = min(32, length)
    # Reshape input into smaller chunks
    s = x.reshape((chunk_size, -1))
    N = s.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Generate a matrix with all exponential values
    M = np.exp(-2j * np.pi * k * n / N)
    # Compute the multiplication of the two matrices
    X = np.dot(M, s)
    # Compute each chunk until result is complete
    while X.shape[0] < length:
        # Separate even and odd number matrices
        h = int(X.shape[1] / 2)
        E = X[:, :h]
        O = X[:, h:]
        # Create a matrix with all factors in order to compute at once
        N = X.shape[0]
        m = np.arange(N)
        factor = np.exp(-1j * np.pi * m / N)[:, None]
        # Compute the X(k) and X(k+N/2) at once
        X = np.vstack([E + factor * O, E - factor * O])
    # Flatten the resulting array
    return X.ravel()




    