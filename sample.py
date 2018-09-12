from pprint import pprint
import mc_conv as conv
import numpy as np
import timeit

matrix = np.random.random((16, 8))
kernel = np.random.random((16, 8))

start = timeit.default_timer()
res = matrix
for i in range(8):
    res = conv.fftconv2d(res, kernel, "same")
pprint(res.round(0))
end = timeit.default_timer() - start
print(end)

print()

start = timeit.default_timer()
res = matrix
for i in range(8):
    res = conv.conv2d(res, kernel, "same")
pprint(res.round(0))
end = timeit.default_timer() - start
print(end)