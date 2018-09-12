# pylint: disable=C0103,C0200

"""

MC Deep Learning Module

This module provides methods that are used to help the training of Convolutional
Neural Networks using Deep Learning.

Given that this module was implemented for academic purposes, these methods might
not support all values.

Example:
    Here is an example that will be enough for 99% of users:

        >>> import mc_deep_learning as dl
        >>> dl.relu([1,2,3,4])
        >>> dl.softmax([1,2,3,4])
        >>> dl.one_hot_encoding([0.3,0.1,0.6])

Requriements:
    * Numpy ~1.13.3

"""

import numpy as np


def relu(array):
    """
    This method is used to compute the activation function "ReLU", introducing
    non-linearity to the training routine.

    Properties:
        * array : list
            1D array containing the data that needs to be activated.
    """
    x = np.array(array, copy=True)
    x[x < 0] = 0
    return x


def softmax(array):
    """
    This method is used to compute the softmax function that converts logits
    into probabilities.

    Properties:
        * array : list
            1D array containing the data that needs to be softmaxed.
    """
    x = np.array(array, copy=True)
    x = np.exp(x)
    return x / sum(x)


def one_hot_encoding(array):
    """
    This method is used to one-hot-encode the probabilities into a vector able
    to be compared to a dataset label.

    Properties:
        * array : list
            1D array containing the data that needs to be one-hot-encoded.
    """
    x = np.array(array, copy=True)
    m = max(x)
    for i in range(len(x)):
        if x[i] == m:
            x[i] = 1
        else:
            x[i] = 0
    return x

def loss():
    """

    

    """
