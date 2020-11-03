

import numpy as np
from datetime import datetime

loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/wine_divide_trainV2_normV2.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]

input_nodes = x_data.shape[1]
hidden_nodes = 8
output_nodes = t_data.shape[1]


W2 = np.random.rand(input_nodes, hidden_nodes) / np.sqrt(input_nodes/2)
b2 = np.random.rand(hidden_nodes)

W3 = np.random.rand(hidden_nodes, output_nodes) / np.sqrt(hidden_nodes/2)
b3 = np.random.rand(output_nodes)

epochs = 1
steps = len(x_data)

learning_rate = 1e-4

delta_x = 1e-4

def sigmoid(z):
    return 1/(1+np.exp(-z))

def numerical_derivative(f, x):
    grad = np.zeros_like(x)
    delta_x = 1e-4
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        temp_val = x[idx]
        x[idx] = float(temp_val) + delta_x
        fx1 = f(x)

        x[idx] = float(temp_val) - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2*delta_x)
        x[idx] = temp_val
        it.iternext()

    return grad

def loss_func(x_data, t_data):
    Z2 = np.dot(x_data, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    y = A3

    return -(np.sum(t_data * np.log(y+delta_x) + (1-t_data) * np.log(1-y+delta_x)))



def predict(x_data):
    Z2 = np.dot(x_data, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    y = A3

    if y > 0.5:
        return 1
    elif y < 0.5:
        return 0

start_time = datetime.now()

print("Initial W2", W2)
print("Initial b2", b2)
print("Initial W3", W3)
print("Initial b3", b3)
print("Initial loss_val", loss_func(x_data, t_data))

for epoch in range(epochs):

    for step in range(x_data.shape[0]):
        input_data = x_data[step]
        output_data = t_data[step]

        f = lambda W : loss_func(input_data, output_data)

        W2 -= learning_rate * numerical_derivative(f, W2)
        b2 -= learning_rate * numerical_derivative(f, b2)
        W3 -= learning_rate * numerical_derivative(f, W3)
        b3 -= learning_rate * numerical_derivative(f, b3)

        if (step % 400 ==0):
            print("step", step, "loss_val", loss_func(input_data, output_data))

    if (epoch % (int(len(loaded_data/50))) == 0):
        print("epoch", epoch, "loss_val", loss_func(input_data, output_data))

'''
for epoch in range(epochs):

    for step in range(x_data.shape[0]):

        f = lambda W : loss_func(x_data, t_data)

        W2 -= learning_rate * numerical_derivative(f, W2)
        b2 -= learning_rate * numerical_derivative(f, b2)
        W3 -= learning_rate * numerical_derivative(f, W3)
        b3 -= learning_rate * numerical_derivative(f, b3)

        if (step % 10 ==0):
            print("step", step, "loss_val", loss_func(x_data, t_data))

    if (epoch % 5 == 0):
        print("epoch", epoch, "loss_val", loss_func(x_data, t_data))

'''


end_time = datetime.now()
print("Time", end_time - start_time)

loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/wine_divide_testV2_normV2.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]

count = 0
for i in range(len(x_data)):
    real_val = t_data[i]
    predicted_val = predict(x_data[i])

    print("predicted_val", int(predicted_val), "real_val", round(float(real_val)))

    if (round(float(predicted_val)) == round(float(real_val))):
        count = count + 1

print("Predict rate", float(count/len(t_data)))




