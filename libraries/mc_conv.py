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

import libraries.mc_fourier as fourier
import numpy as np


def shift_bit_length(x):
    """
    This method is used to find the next power of two of a given number.

    Properties:
        * x : int
            an integer number from which to look for the next power of
            two number.
    """
    # Convert the number to binary and shift a bit to find the next pow2
    return 1 << (x-1).bit_length()


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
    # Pad to the next power of two
    next_pow2 = shift_bit_length(len(x))
    x = np.pad(
        x,
        ((0, next_pow2 - len(x)),
         (0, next_pow2 - len(x))),
        mode='constant'
    )
    y = np.pad(
        y,
        ((0, next_pow2 - len(y)),
         (0, next_pow2 - len(y))),
        mode='constant'
    )
    # Flatten the matrices before computing the transform
    x = x.reshape(-1)
    y = y.reshape(-1)
    # Compute the transforms
    x = fourier.fft(x)
    y = fourier.fft(y)
    # Compute the Hadamard product
    z = x*y
    # Compute the inverse transform
    z = np.fft.ifft(z).real
    # Reshape the output into a matrix
    z = z.reshape((next_pow2, next_pow2))
    # Get only the correct output size
    z = z[:z_final_shape[0], :z_final_shape[1]]
    # Return the correct output slice as requested
    if mode == "same":
        return trim_edges(z, x_initial_shape)
    elif mode == "valid":
        return trim_edges(z, x_initial_shape - y_initial_shape + 1)
    return z


def multiple_fftconv2d(images, kernels, mode="valid"):
    """
    This method is used to compute a convolution between multiple 2D
    matrices usign Fourier Transform. Supports modes: full, same, valid.

    Properties:
        * images : list
            3D array containing all matrices that will suffer the
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
    images = [np.asarray(image, dtype=float) for image in images]
    kernels = [np.asarray(kernel, dtype=float) for kernel in kernels]
    # Calculate output shape
    y_total_shape = [0, 0]
    x_initial_shape = np.array(images[0].shape)
    z_real_final_shape = x_initial_shape
    for kernel in kernels:
        y_total_shape += np.array(kernel.shape) - 1
        z_real_final_shape = z_real_final_shape + np.array(kernel.shape) - 1
    next_pow2 = shift_bit_length(int(z_real_final_shape[0]))
    z_final_shape = (next_pow2, next_pow2)
    # Pad the input to the same size as the final shape
    images = [
        np.pad(
            image,
            ((0, z_final_shape[0]-image.shape[0]),
             (0, z_final_shape[1]-image.shape[1])),
            mode='constant'
        )
        for image in images
    ]
    # Pad the kernels to the same size as the input
    kernels = [
        np.pad(
            kernel,
            ((0, z_final_shape[0]-kernel.shape[0]),
             (0, z_final_shape[1]-kernel.shape[1])),
            mode='constant'
        )
        for kernel in kernels
    ]
    # Flatten the matrices before computing the transform
    images = [image.reshape(-1) for image in images]
    kernels = [kernel.reshape(-1) for kernel in kernels]
    # Compute the transforms
    images = [fourier.fft(image) for image in images]
    kernels = [fourier.fft(kernel) for kernel in kernels]
    # Compute the Hadamard product
    for image in images:
        z = image
        for kernel in kernels:
            z *= kernel
    # Compute the inverse transform
    z = np.fft.ifft(z).real
    # Reshape the output into a matrix
    z = z.reshape(z_final_shape)
    # Remove power of two padding
    z = z[:z_real_final_shape[0], :z_real_final_shape[1]]
    # Return the correct output slice as requested
    if mode == "same":
        return trim_edges(z, x_initial_shape)
    elif mode == "valid":
        return trim_edges(z, x_initial_shape - y_total_shape)
    return z


def custom_multiple_fftconv2d(images, kernels, fft, ifft, mode="valid"):
    """
    This method is used to compute a convolution between multiple 2D
    matrices usign Fourier Transform. Supports modes: full, same, valid.

    Properties:
        * images : list
            3D array containing all matrices that will suffer the
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
    images = [np.asarray(image, dtype=float) for image in images]
    kernels = [np.asarray(kernel, dtype=float) for kernel in kernels]
    # Calculate output shape
    y_total_shape = [0, 0]
    x_initial_shape = np.array(images[0].shape)
    z_real_final_shape = x_initial_shape
    for kernel in kernels:
        y_total_shape += np.array(kernel.shape) - 1
        z_real_final_shape = z_real_final_shape + np.array(kernel.shape) - 1
    next_pow2 = shift_bit_length(int(z_real_final_shape[0]))
    z_final_shape = (next_pow2, next_pow2)
    # Pad the input to the same size as the final shape
    images = [
        np.pad(
            image,
            ((0, z_final_shape[0]-image.shape[0]),
             (0, z_final_shape[1]-image.shape[1])),
            mode='constant'
        )
        for image in images
    ]
    # Pad the kernels to the same size as the input
    kernels = [
        np.pad(
            kernel,
            ((0, z_final_shape[0]-kernel.shape[0]),
             (0, z_final_shape[1]-kernel.shape[1])),
            mode='constant'
        )
        for kernel in kernels
    ]
    # Flatten the matrices before computing the transform
    images = [image.reshape(-1) for image in images]
    kernels = [kernel.reshape(-1) for kernel in kernels]
    # Compute the transforms
    images = [fft(image) for image in images]
    kernels = [fft(kernel) for kernel in kernels]
    # Compute the Hadamard product
    for image in images:
        z = image
        for kernel in kernels:
            z *= kernel
    # Compute the inverse transform
    z = ifft(z).real
    # Reshape the output into a matrix
    z = z.reshape(z_final_shape)
    # Remove power of two padding
    z = z[:z_real_final_shape[0], :z_real_final_shape[1]]
    # Return the correct output slice as requested
    if mode == "same":
        return trim_edges(z, x_initial_shape)
    elif mode == "valid":
        return trim_edges(z, x_initial_shape - y_total_shape)
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
    # Convert data to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Calculate output shape
    s1 = np.array(x.shape)
    s2 = np.array(y.shape)
    shape = s1 + s2 - 1
    # Create the output matrix
    ret = np.zeros(shape)
    # Calculate the diference between current and output shapes
    shape_diff = shape - s1
    # Slide the kernel over the input, computing matrix multiplications
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
    # Trim the edges of the output matrix to get the correct padding mode
    if mode == "same":
        return trim_edges(ret, s1)
    elif mode == "valid":
        return trim_edges(ret, s1 - s2 + 1)
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
    # Turn the shapes into arrays for easy computation
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    # Calculate the start and end coordenates for the output (middle)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    # Define a matrix slice using the coordenates
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    # Return the requested slice of the input matrix
    return arr[tuple(myslice)]
