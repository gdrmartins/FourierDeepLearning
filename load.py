"""
Load the MNIST dataset into numpy arrays
Author: Alexandre Drouin
License: BSD
"""
import numpy as np

# from tensorflow.examples.tutorials.mnist import input_data


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

# x_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
# y_train = mnist.train.labels

# del mnist

# x_train = [img.reshape(28, 28) for img in x_train][:100]

images = np.load('mnist_inputs.npy')[:100]

np.save('mnist_inputs_lenet.npy', images)