import mc_conv as conv
import mc_fourier as fourier
import numpy as np
from pprint import pprint
import timeit


def display(res):
    res = [['{:05.2F}'.format(np.around(res[j][i], decimals=2))
            for i in range(res.shape[0])] for j in range(res.shape[1])]
    pprint(res)


matrix = np.random.random((64, 64))
kernels = []
kernels.append(np.random.random((8, 8)))
kernels.append(np.random.random((8, 8)))
kernels.append(np.random.random((8, 8)))
kernels.append(np.random.random((8, 8)))
kernels.append(np.random.random((8, 8)))
kernels.append(np.random.random((8, 8)))


start = timeit.default_timer()

print()
res = matrix
for kernel in kernels:
    res = conv.conv2d(res, kernel, "valid")
display(res)

print(timeit.default_timer() - start)
start = timeit.default_timer()

print()
res = matrix
for kernel in kernels:
    res = conv.fftconv2d(res, kernel, "valid")
display(res)

print(timeit.default_timer() - start)
start = timeit.default_timer()

print()
res = matrix
res = conv.multiple_fftconv2d(res, kernels, "valid")
display(res)

print(timeit.default_timer() - start)