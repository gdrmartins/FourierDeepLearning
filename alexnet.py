###################################################################################################
######################################### ALEXNET BENCHMARK #######################################
###################################################################################################

# CUSTOM LIBRARIES
import libraries.mc_conv as conv

# MATHEMATICAL LIBRARIES
import numpy as np
from scipy import signal, fftpack

# HELPER LIBRARIES
import os
import timeit
from pprint import pprint

###################################################################################################
######################################### HELPER FUNCTIONS ########################################
###################################################################################################

def display(name, res, time):

	print(name, time)

	shape = res.shape if max(res.shape) < 6 else (5, 5)
	res = [['{:06.2F}'.format(np.around(res[j][i], decimals=2))
		for i in range(shape[0])] for j in range(shape[1])]
	pprint(res)

	f = open("times.csv", "a")
	f.write(name + ";" + str(time) + "\n")

def break_file():
	f = open("times.csv", "a")
	f.write("\n")

if os.path.exists("times.csv"):
  os.remove("times.csv")
for _ in range(10):

	###################################################################################################
	####################################### NETWORK ARCHITECTURE ######################################
	###################################################################################################

	images = np.load('mnist_inputs_alexnet.npy')[:100]
	
	kernels = []
	kernels.append((np.zeros((11, 11))+0.1))
	kernels.append((np.zeros((5, 5))+0.1))
	kernels.append((np.zeros((3, 3))+0.1))
	kernels.append((np.zeros((3, 3))+0.1))
	kernels.append((np.zeros((3, 3))+0.1))

	###################################################################################################
	###################################### SLIDING-WINDOW CONVOLVE ####################################
	###################################################################################################

	# start = timeit.default_timer()
	# print()
	# for image in images:
	# 	res = image
	# 	for kernel in kernels:
	# 		res = conv.conv2d(res, kernel, mode="valid")
	# display("Sliding-Window Convolve:", res, timeit.default_timer() - start)
	break_file()

	###################################################################################################
	########################################## FOURIER CONVOLVE #######################################
	###################################################################################################

	start = timeit.default_timer()
	print()
	for image in images:
		res = image
		for kernel in kernels:
			res = conv.fftconv2d(res, kernel, mode="valid")
	display("Fourier Convolve:", res, timeit.default_timer() - start)

	###################################################################################################
	####################################### FAST FOURIER CONVOLVE #####################################
	###################################################################################################

	start = timeit.default_timer()
	print()
	results = conv.multiple_fftconv2d(images, kernels, mode="valid")
	res1 = np.array(results, copy=True)
	display("Fast Fourier Convolve:", res1, timeit.default_timer() - start)

	###################################################################################################
	########################################## SCIPY CONVOLVE #########################################
	###################################################################################################

	start = timeit.default_timer()
	print()
	for image in images:
		res = image
		for kernel in kernels:
			res = signal.convolve2d(res, kernel, mode="valid")
	res2 = np.array(res, copy=True)
	display("Scipy Convolve:", res2, timeit.default_timer() - start)

	###################################################################################################
	###################################### SCIPY FOURIER CONVOLVE #####################################
	###################################################################################################

	start = timeit.default_timer()
	print()
	for image in images:
		res = image
		for kernel in kernels:
			res = signal.fftconvolve(res, kernel, mode="valid")
	display("Scipy Fourier Convolve:", res, timeit.default_timer() - start)

	###################################################################################################
	################################### SCIPY FAST FOURIER CONVOLVE ###################################
	###################################################################################################

	start = timeit.default_timer()
	print()
	results = conv.custom_multiple_fftconv2d(images, kernels, fftpack.fft, fftpack.ifft, mode="valid")
	display("Scipy Fast Fourier Convolve:", results, timeit.default_timer() - start)

	###################################################################################################
	######################################### ERROR BENCHMARK #########################################
	###################################################################################################

	print("\nError:", sum([sum(i) for i in abs(res1 - res2)]))

	###################################################################################################
	########################################### END OF FILE ###########################################
	###################################################################################################
	
	break_file()