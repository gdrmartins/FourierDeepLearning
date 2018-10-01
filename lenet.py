###################################################################################################
########################################## LENET BENCHMARK ########################################
###################################################################################################

# CUSTOM LIBRARIES
import libraries.mc_conv as conv

# MATHEMATICAL LIBRARIES
import numpy as np
from scipy import signal, fftpack

# HELPER LIBRARIES
import timeit
from pprint import pprint

###################################################################################################
######################################### HELPER FUNCTIONS ########################################
###################################################################################################

def display(res):
    shape = res.shape if max(res.shape) < 6 else (5, 5)
    res = [['{:07.2F}'.format(np.around(res[j][i], decimals=2))
            for i in range(shape[0])] for j in range(shape[1])]
    pprint(res)

###################################################################################################
####################################### NETWORK ARCHITECTURE ######################################
###################################################################################################

images = []
for _ in range(100):
    images.append(np.random.normal(0.25, 0.25, (32, 32)))

kernels = []
kernels.append(np.random.normal(0.25, 0.25, (5, 5)))
kernels.append(np.random.normal(0.25, 0.25, (15, 15)))
kernels.append(np.random.normal(0.25, 0.25, (5, 5)))
kernels.append(np.random.normal(0.25, 0.25, (6, 6)))

###################################################################################################
###################################### SLIDING-WINDOW CONVOLVE ####################################
###################################################################################################

start = timeit.default_timer()
print()
for image in images:
    res = image
    for kernel in kernels:
        res = conv.conv2d(res, kernel, mode="valid")
display(res)
print("Sliding-Window Convolve:", timeit.default_timer() - start)

###################################################################################################
########################################## FOURIER CONVOLVE #######################################
###################################################################################################

start = timeit.default_timer()
print()
for image in images:
    res = image
    for kernel in kernels:
        res = conv.fftconv2d(res, kernel, mode="valid")
display(res)
print("Fourier Convolve:", timeit.default_timer() - start)

###################################################################################################
####################################### FAST FOURIER CONVOLVE #####################################
###################################################################################################

start = timeit.default_timer()
print()
results = conv.multiple_fftconv2d(images, kernels, mode="valid")
display(res)
res1 = np.array(results, copy=True)
print("Fast Fourier Convolve:", timeit.default_timer() - start)

###################################################################################################
########################################## SCIPY CONVOLVE #########################################
###################################################################################################

start = timeit.default_timer()
print()
for image in images:
    res = image
    for kernel in kernels:
        res = signal.convolve2d(res, kernel, mode="valid")
display(res)
res2 = np.array(res, copy=True)
print("Scipy Convolve:", timeit.default_timer() - start)

###################################################################################################
###################################### SCIPY FOURIER CONVOLVE #####################################
###################################################################################################

start = timeit.default_timer()
print()
for image in images:
    res = image
    for kernel in kernels:
        res = signal.fftconvolve(res, kernel, mode="valid")
display(res)
print("Scipy Fourier Convolve:", timeit.default_timer() - start)

###################################################################################################
################################### SCIPY FAST FOURIER CONVOLVE ###################################
###################################################################################################

start = timeit.default_timer()
print()
results = conv.custom_multiple_fftconv2d(images, kernels, fftpack.fft, fftpack.ifft, mode="valid")
display(res)
print("Scipy Fast Fourier Convolve:", timeit.default_timer() - start)

###################################################################################################
######################################### ERROR BENCHMARK #########################################
###################################################################################################

print("\nError:", sum([sum(i) for i in abs(res1 - res2)]))

###################################################################################################
########################################### END OF FILE ###########################################
###################################################################################################