import mc_conv as conv
import mc_fourier as fourier
import numpy as np
from pprint import pprint
import timeit
from scipy import signal


def display(res):
    shape = res.shape
    res = [['{:05.2F}'.format(np.around(res[j][i], decimals=2))
            for i in range(shape[0])] for j in range(shape[1])]
    pprint(res)


images = []
for _ in range(100):
    images.append(np.random.normal(0.05, 0.01, (224, 224)))

kernels = []
kernels.append(np.random.normal(0.05, 0.01, (11, 11)))
kernels.append(np.random.normal(0.05, 0.01, (5, 5)))
kernels.append(np.random.normal(0.05, 0.01, (3, 3)))
kernels.append(np.random.normal(0.05, 0.01, (3, 3)))
kernels.append(np.random.normal(0.05, 0.01, (3, 3)))

# start = timeit.default_timer()
# print()
# for image in images:
#     res = image
#     for kernel in kernels:
#         res = conv.conv2d(res, kernel, "valid")
#     # display(res)
# print("Sliding-Window Convolve:", timeit.default_timer() - start)

start = timeit.default_timer()
print()
for image in images:
    res = image
    for kernel in kernels:
        res = conv.fftconv2d(res, kernel, "valid")
    # display(res)
print("Fourier Convolve:", timeit.default_timer() - start)

start = timeit.default_timer()
print()
results = conv.multiple_fftconv2d(images, kernels, "valid")
# for res in results:
#     display(res)
res1 = np.array(results, copy=True)
print("Fast Fourier Convolve:", timeit.default_timer() - start)

start = timeit.default_timer()
print()
for image in images:
    res = image
    for kernel in kernels:
        res = signal.convolve2d(res, kernel, "valid")
    # display(res)
res2 = np.array(res, copy=True)
print("Scipy Convolve:", timeit.default_timer() - start)

start = timeit.default_timer()
print()
for image in images:
    res = image
    for kernel in kernels:
        res = signal.fftconvolve(res, kernel, "valid")
    # display(res)
print("Scipy Fourier Convolve:", timeit.default_timer() - start)

start = timeit.default_timer()
print()
results = conv.scipy_multiple_fftconv2d(images, kernels, "valid")
# for res in results:
#     display(res)
print("Scipy Fast Fourier Convolve:", timeit.default_timer() - start)


print("\nError:", sum([sum(i) for i in abs(res1 - res2)]))