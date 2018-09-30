#pylint: disable=C0103

"""

MC Convolution Module

This module provides methods that are used to compute the convolution
between matrixes.

Given that this module was implemented for academic purposes, this
convolution implementation supports only 2D vectors.

Example:
    Here is an example that will be enough for 99% of users:

        >>> import mc_conv as conv
        >>> conv.conv2d([[0, 1, 2],[0, 1, 2],[0, 1, 2]],[[1, 2],[1, 2]], "full")

Requriements:
    * Numpy ~1.13.3

"""

import mc_fourier as fourier
import numpy as np
import timeit


def shift_bit_length(x):
    return 1<<(x-1).bit_length()


def fftconv2d(x, y, mode="valid"):
    """
    This method is used to compute a convolution between two 2D
    matrix usign Fourier Transform. Supports modes: full, same, valid.

    Properties:
        * x : list
            2D array containing the matrix that will suffer the
            convolution.
        * y : list
            2D array containing the matrix that will serve as the
            kernel "filter" for the convolution.
        * mode : string
            string indicating the mode of convolution, or, in other
            words, the size of the output matrix. {full, same, valid}
    """
    # Convert data to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Calculate output shape
    x_initial_shape = np.array(x.shape)
    y_initial_shape = np.array(y.shape)
    z_final_shape = x_initial_shape + y_initial_shape - 1
    # Calculate the padding needed to make y the same as x
    colpad = (0, x.shape[0]-y.shape[0])
    rowpad = (0, x.shape[1]-y.shape[1])
    # Pad the kernel to the same size as the input
    y = np.pad(y, (colpad, rowpad), mode='constant')
    # Pad the matrices to allow for a linear convolution to take place
    nrows, ncols = x.shape
    x = np.pad(x, ((0, nrows-1), (0, ncols-1)), mode='constant')
    y = np.pad(y, ((0, nrows-1), (0, ncols-1)), mode='constant')
    padded_shape = x.shape
    # Compute the transforms
    x = fourier.fft(x)
    y = fourier.fft(y)
    # Flatten the matrices before computing the transform
    x = x.reshape(-1)
    y = y.reshape(-1)
    # Compute the Hadamard product
    z = x*y
    # Compute the inverse transform
    z = np.fft.ifft(z).real
    # Reshape the output into a matrix
    z = z.reshape(padded_shape)
    # Get only the correct output size
    z = z[:z_final_shape[0], :z_final_shape[1]]
    # Return the correct output slice as requested
    if mode == "same":
        return trim_edges(z, np.array(x_initial_shape))
    elif mode == "valid":
        return trim_edges(z, np.array(x_initial_shape) - np.array(y_initial_shape) + 1)
    return z


def multiple_fftconv2d(input, kernels, epochs, mode="valid"):
    """
    This method is used to compute a convolution between multiple 2D
    matrices usign Fourier Transform. Supports modes: full, same, valid.

    Properties:
        * input : list
            2D array containing the matrix that will suffer the
            convolution.
        * kernels : list
            3D array containing all matrices that will serve as the
            kernel "filters" for the convolution.
        * epochs : list
            1D array containing the epochs to run the convolutions
        * mode : string
            string indicating the mode of convolution, or, in other
            words, the size of the output matrix. {full, same, valid}
    """
    # Convert data to numpy arrays
    x = np.asarray(input, dtype=float)
    kernels = [np.asarray(kernel, dtype=float) for kernel in kernels]
    # Calculate output shape
    x_initial_shape = np.array(x.shape)
    z_real_final_shape = x_initial_shape
    for kernel in kernels:
        z_real_final_shape = z_real_final_shape + np.array(kernel.shape) - 1
    next_pow2 = shift_bit_length(int(z_real_final_shape[0]))
    z_final_shape = (next_pow2, next_pow2)
    # Pad the input to the same size as the final shape
    x = np.pad(x, ((0, z_final_shape[0]-x.shape[0]), (0, z_final_shape[1]-x.shape[1])), mode='constant')
    # Pad the kernels to the same size as the input
    kernels = [np.pad(kernel, ((0, x.shape[0]-kernel.shape[0]), (0, x.shape[1]-kernel.shape[1])), mode='constant') for kernel in kernels]
    # Flatten the matrices before computing the transform
    x = x.reshape(-1)
    kernels = [kernel.reshape(-1) for kernel in kernels]
    # Compute the transforms
    start = timeit.default_timer()
    x = fourier.fft(x)
    kernels = [fourier.fft(kernel) for kernel in kernels]
    print("Multi-Fourier-Conv2d pre-processing:", timeit.default_timer() - start)
    # Compute the Hadamard product
    z = x
    start = timeit.default_timer()
    for _ in epochs:
        for kernel in kernels:
            z *= kernel
    print("Multi-Fourier-Conv2d pos-processing:", timeit.default_timer() - start)
    # Compute the inverse transform
    z = np.fft.ifft(z).real
    # Reshape the output into a matrix
    z = z.reshape(z_final_shape)

    # Remove power of two padding
    z = z[:z_real_final_shape[0],:z_real_final_shape[1]]
    # Return the correct output slice as requested
    if mode == "same":
        return trim_edges(z, x_initial_shape)
    elif mode == "valid":
        valid_shape = x_initial_shape - abs(z_real_final_shape - x_initial_shape)
        return trim_edges(z, valid_shape)
    return z


def conv2d(x, y, mode="valid"):
    """
    This method is used to compute a convolution between two 2D
    matrix using "Sliding Window". Supports modes: full, same, valid.

    Properties:
        * x : list
            2D array containing the matrix that will suffer the
            convolution.
        * y : list
            2D array containing the matrix that will serve as the
            kernel "filter" for the convolution.
        * mode : string
            string indicating the mode of convolution, or, in other
            words, the size of the output matrix. {full, same, valid}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    s1 = np.array(x.shape)
    s2 = np.array(y.shape)

    shape = s1 + s2 - 1

    ret = np.zeros(shape)
    shape_diff = shape - s1

    for i in range(shape[0]):
        for j in range(shape[1]):
            value = 0
            for k in range(s2[0]):
                for l in range(s2[1]):
                    iaux = i - shape_diff[0] + k
                    jaux = j - shape_diff[1] + l
                    if iaux >= 0 and jaux >= 0 and iaux < x.shape[0] and jaux < x.shape[1]:
                        value += y[-1-k][-1-l] * x[iaux][jaux]
            ret[i][j] = value

    if mode == "same":
        return trim_edges(ret, np.array(x.shape))
    elif mode == "valid":
        return trim_edges(ret, np.array(x.shape) - np.array(y.shape) + 1)
    return ret


def trim_edges(arr, newshape):
    """
    Trim the edges of a 2D matrix until the desired shape is established.

    Properties:
        * arr : list
            2D array containing the matrix that needs to be trimmed.
        * newshape : tuple
            A 2-index tuple with the length of each of the 2 dimensions.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
