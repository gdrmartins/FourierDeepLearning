# pylint: skip-file

import mark_deep_learning as dl
import mark_fourier as fourier
import mark_conv as conv
import numpy as np

np.random.seed(598765)

x1 = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
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

    layer_1 = conv.conv2d(X[i], w1)
    layer_1_act = dl.relu(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = dl.relu(layer_2)
    cost = np.square(layer_2_act - Y[i]).sum() * 0.5
    cost_before_train = cost_before_train + cost
    start_out = np.append(start_out, layer_2_act)

# ----- TRAINING -------
for iter in range(num_epoch):

    for i in range(len(X)):

        layer_1 = conv.conv2d(X[i], w1)
        layer_1_act = dl.relu(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = dl.relu(layer_2)

        cost = np.square(layer_2_act - Y[i]).sum() * 0.5
        #print("Current iter : ",iter , " Current train: ",i, " Current cost: ",cost,end="\r")

        grad_2_part_1 = layer_2_act - Y[i]
        grad_2_part_2 = dl.relu(layer_2)
        grad_2_part_3 = layer_1_act_vec
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = dl.relu(layer_1)
        grad_1_part_3 = X[i]

        grad_1_part_1_reshape = np.reshape(grad_1_part_1, (2, 2))
        grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
        grad_1 = np.rot90(
            conv.conv2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2)),
            2)

        w2 = w2 - grad_2 * learning_rate
        w1 = w1 - grad_1 * learning_rate

# ---- Cost after training ------
for i in range(len(X)):

    layer_1 = conv.conv2d(X[i], w1)
    layer_1_act = dl.relu(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = dl.relu(layer_2)
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
