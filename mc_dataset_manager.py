# pylint: disable=C0103,C0200

"""

MC Dataset Manager Module

This module provides methods that are used to help the pre-processing of the MNIST
dataset for later training.

Given that this module was implemented for academic purposes, these methods might
not support all values.

Example:
    Here is an example that will be enough for 99% of users:

        >>> import mc_dataset_manager as dm
        >>> dm.download()
        >>> dm.load_dataset()
        >>> dm.get_batch(50)

Requriements:
    * Numpy ~1.13.3

"""

import os
import random

import numpy as np
import mc_fourier as fourier
import mnist_to_jpg as mnist2jpg

TRAIN_DATA = []
TRAIN_LABELS = []
TEST_DATA = []
TEST_LABELS = []

def download():
    """
    This method is used to easely download the MNIST dataset, extract
    it and save as numpy arrays in a "data" folder.
    """
    # Download MNIST datasets and labels
    train_data_filename = mnist2jpg.maybe_download(
        'train-images-idx3-ubyte.gz')
    train_labels_filename = mnist2jpg.maybe_download(
        'train-labels-idx1-ubyte.gz')
    test_data_filename = mnist2jpg.maybe_download(
        't10k-images-idx3-ubyte.gz')
    test_labels_filename = mnist2jpg.maybe_download(
        't10k-labels-idx1-ubyte.gz')
    # Extract MNIST datasets and labels
    train_data = mnist2jpg.extract_data(train_data_filename, 60000)
    train_labels = mnist2jpg.extract_labels(train_labels_filename, 60000)
    test_data = mnist2jpg.extract_data(test_data_filename, 10000)
    test_labels = mnist2jpg.extract_labels(test_labels_filename, 10000)
    # Delete temporary data
    for file in os.listdir("./data"):
        os.remove("./data/"+file)
    # Normalize dataset
    train_data = train_data.reshape(60000,784)
    test_data = test_data.reshape(10000,784)
    train_labels = (np.arange(10) == train_labels[:,None]).astype(np.float32)
    test_labels = (np.arange(10) == test_labels[:,None]).astype(np.float32)
    datasets = [train_data, test_data]
    for i, dataset in enumerate(datasets):
        for j, image in enumerate(dataset):
            for k, value in enumerate(image):
                datasets[i][j][k] = (value-128)/128
    # Save MNIST datasets and labels as .npy files
    np.save('./data/train_data.npy', train_data)
    np.save('./data/train_labels.npy', train_labels)
    np.save('./data/test_data.npy', test_data)
    np.save('./data/test_labels.npy', test_labels)

def load_dataset():
    """
    This method is used to load a previously downloaded MNIST dataset
    stored in the "data" folder as numpy arrays.
    """
    global TRAIN_DATA
    global TRAIN_LABELS
    global TEST_DATA
    global TEST_LABELS

    TRAIN_DATA = np.load('./data/train_data.npy')
    TRAIN_LABELS = np.load('./data/train_labels.npy')
    TEST_DATA = np.load('./data/test_data.npy')
    TEST_LABELS = np.load('./data/test_labels.npy')

def get_batch(size, test=False):
    """
    This method is used to get a single batch of data from the MNIST
    training/test dataset.

    Properties:
        * size : int
            The number of samples to return from the MNIST dataset
    """
    batch = []
    dataset = TEST_DATA if test else TRAIN_DATA
    labels = TEST_LABELS if test else TRAIN_LABELS
    for _ in range(size):
        rn = random.randrange(0, len(dataset))
        batch.append([dataset[rn], labels[rn]])
    return np.array(batch)

#download()
load_dataset()
print(get_batch(50)[0][0].shape)