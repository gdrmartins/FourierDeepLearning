# pylint: disable=C0103,W0612

"""

MC Trainer Module

This module trains a deep convolutional neural network to recognize hand-
written natural numbers using fourier convolution with the MNIST dataset.

Given that this module was implemented for academic purposes, this trainer
might not support other models, datasets or parameters.

Requriements:
    * Numpy ~1.13.3

"""

import numpy as np


def mse(x, y):
    """
    todo
    """
    return ((x-y)**2).mean(axis=1)


def compute_cost(result, label):
    """
    This method computes the cost, or distance, between the expected result
    and the actual result of the neural network.
    """

    # cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))
    return result


def train():
    """
    This method trains a deep convolutional neural network to recognize
    hand-written natural numbers using fourier convolution with the MNIST
    dataset.
    """

    learning_rate = 0.01
    batch_size = 50
    epochs = 1000

    layer_1 = []
    bias_1 = []

    layer_2 = []
    bias_2 = []

    layer_3 = []
    bias_3 = []

    layer_4 = []
    bias_4 = []

    layer_5 = []
    bias_5 = []

    layer_6 = []
    bias_6 = []

    return np.zeros((2, 2))

# train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
# train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
# test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
# test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

# train_data = extract_data(train_data_filename, 60000)
# train_labels = extract_labels(train_labels_filename, 60000)
# test_data = extract_data(test_data_filename, 10000)
# test_labels = extract_labels(test_labels_filename, 10000)