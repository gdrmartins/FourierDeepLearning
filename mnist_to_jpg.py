"""

Imanol Schlag's MNIST to JPG Converter

Simple python script which takes the mnist data from tensorflow and builds a data
set based on jpg files and text files containing the image paths and labels. Parts
of it are from the mnist tensorflow example.

Example:
    >>> import mnist_to_jpg as mnist2jpg

    >>> train_data_filename = mnist2jpg.maybe_download('train-images-idx3-ubyte.gz')
    >>> train_labels_filename = mnist2jpg.maybe_download('train-labels-idx1-ubyte.gz')
    >>> test_data_filename = mnist2jpg.maybe_download('t10k-images-idx3-ubyte.gz')
    >>> test_labels_filename = mnist2jpg.maybe_download('t10k-labels-idx1-ubyte.gz')

    >>> train_data = mnist2jpg.extract_data(train_data_filename, 60000)
    >>> train_labels = mnist2jpg.extract_labels(train_labels_filename, 60000)
    >>> test_data = mnist2jpg.extract_data(test_data_filename, 10000)
    >>> test_labels = mnist2jpg.extract_labels(test_labels_filename, 10000)

Requriements:
    * Tensorflow ~1.8.0
    * Numpy ~1.13.3

Credits:
    * Imanol Schlag (https://gist.github.com/ischlag)

"""
import os
import gzip

import tensorflow as tf
import numpy as np
import urllib.request

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def maybe_download(filename):
    """
    Download the data from Yann's website, unless it's already here.
    """
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(
            SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def extract_data(filename, num_images):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data


def extract_labels(filename, num_images):
    """
    Extract the labels into a vector of int64 label IDs.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
