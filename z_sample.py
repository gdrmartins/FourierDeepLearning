import mc_conv as conv
import mc_fourier as fourier
import numpy as np
from pprint import pprint
import timeit
from scipy import signal


def display(res):
    shape = (8,8)#res.shape
    res = [['{:05.2F}'.format(np.around(res[j][i], decimals=2))
            for i in range(shape[0])] for j in range(shape[1])]
    pprint(res)


matrix = np.random.normal(0.05, 0.01, (320, 320))
kernels = []
kernels.append(np.random.normal(0.05, 0.01, (28, 28)))
# kernels.append(np.random.normal(0.05, 0.01, (14, 14)))
# kernels.append(np.random.normal(0.05, 0.01, (10, 10)))
# kernels.append(np.random.normal(0.05, 0.01, (5, 5)))

epochs = range(1)

start = timeit.default_timer()

print()
res = matrix
for _ in epochs:
    for kernel in kernels:
        res = signal.fftconvolve(res, kernel, "same")
display(res)

res1 = np.array(res, copy=True)

print("Scipy FFT Convolve:", timeit.default_timer() - start)
start = timeit.default_timer()

print()
res = matrix
res = conv.multiple_fftconv2d(res, kernels, epochs, "same")
display(res)

print("Multi-Fourier-Conv2d:", timeit.default_timer() - start)

print("\nError:", sum([sum(i) for i in abs(res1 - res)]))