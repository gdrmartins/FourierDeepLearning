import numpy as np

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def log(x):
    return 1 / (1 + np.exp(-1 * x))

def d_log(x):
    return log(x) * (1 - log(x))

def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def fft_convolve2d(x, y):
    """ 2D convolution, using FFT"""

    shape = y.shape
    size = y.size

    x = np.reshape(x, -1)
    y = np.reshape(y, -1)

    y = np.pad(y, [(0, x.shape[0] - y.shape[0])], mode='constant')


    fx = np.fft.fft(x)
    fy = np.fft.fft(y)

    n = fx.size

    conv = np.convolve(fx, np.concatenate((fy, fy)))
    conv = conv[n:2*n]

    reverse = np.fft.ifft(conv)/n

    return np.reshape(reverse[:size], shape)

    # fr = np.fft.fft2(x)
    # fr2 = np.fft.fft2(np.flipud(np.fliplr(y)))
    # m, n = fr.shape
    # cc = np.real(np.fft.ifft2(fr * fr2))
    # cc = np.roll(cc, int(-m / 2 + 1), axis=0)
    # cc = np.roll(cc, int(-n / 2 + 1), axis=1)
    # return cc


np.random.seed(598765)

x1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
x2 = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
x3 = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
x4 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

X = np.array([x1, x2, x3, x4])
Y = np.array([0.53, 0.77, 0.88, 1.10])

w1 = np.random.randn(2, 2) * 4
w2 = np.random.randn(4, 1) * 4

num_epoch = 1000
learning_rate = 0.7

cost_before_train = 0
cost_after_train = 0
final_out, start_out = np.array([[]]), np.array([[]])

# ---- Cost before training ------
for i in range(len(X)):

    layer_1 = conv2d(X[i], w1)
    layer_1_act = tanh(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = log(layer_2)
    cost = np.square(layer_2_act - Y[i]).sum() * 0.5
    cost_before_train = cost_before_train + cost
    start_out = np.append(start_out, layer_2_act)

# ----- TRAINING -------
for iter in range(num_epoch):

    for i in range(len(X)):

        layer_1 = conv2d(X[i], w1)
        layer_1_act = tanh(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = log(layer_2)

        cost = np.square(layer_2_act - Y[i]).sum() * 0.5
        #print("Current iter : ",iter , " Current train: ",i, " Current cost: ",cost,end="\r")

        grad_2_part_1 = layer_2_act - Y[i]
        grad_2_part_2 = d_log(layer_2)
        grad_2_part_3 = layer_1_act_vec
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_tanh(layer_1)
        grad_1_part_3 = X[i]

        grad_1_part_1_reshape = np.reshape(grad_1_part_1, (2, 2))
        grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
        grad_1 = np.rot90(
            conv2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2)),
            2)

        w2 = w2 - grad_2 * learning_rate
        w1 = w1 - grad_1 * learning_rate

# ---- Cost after training ------
for i in range(len(X)):

    layer_1 = conv2d(X[i], w1)
    layer_1_act = tanh(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = log(layer_2)
    cost = np.square(layer_2_act - Y[i]).sum() * 0.5
    cost_after_train = cost_after_train + cost
    final_out = np.append(final_out, layer_2_act)


# ----- Print Results ---
print("\nW1 :", w1, "\n\nw2 :", w2)
print("----------------")
print("Cost before Training: ", cost_before_train)
print("Cost after Training: ", cost_after_train)
print("----------------")
print("Start Out put : ", start_out)
print("Final Out put : ", final_out)
print("Ground Truth  : ", Y.T)

# -- end code --
