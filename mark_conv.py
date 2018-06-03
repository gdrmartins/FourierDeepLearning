#pylint: disable=C0103

"""

Mark Convolution Module

This module provides methods that are used to compute the convolution
between matrixes.

Given that this module was implemented for academic purposes, this
convolution implementation supports only 2D vectors.

Example:
    Here is an example that will be enough for 99% of users:

        >>> import mark_conv as conv
        >>> conv.conv2d([[0, 1, 2],[0, 1, 2],[0, 1, 2]],[[1, 2],[1, 2]], "full")

Requriements:
    * Numpy ~1.13.3

"""

import numpy as np

def conv2d(x, y, mode="valid"):
    """
    This method is used to compute a convolution between two 2D
    matrix. Supports modes: full, same, valid.

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
