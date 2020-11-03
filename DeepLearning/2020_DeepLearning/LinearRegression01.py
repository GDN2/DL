import numpy as np
import os

print(os.getcwd())
loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/data-01.csv', delimiter=',',
                         dtype=np.float32)

x_data = loaded_data[:, :-1]
t_data = loaded_data[:, [-1]]

print("loaded_data.shape", loaded_data.shape)
print("x_data.shape", x_data.shape)
print("t_data.shape", t_data.shape)

W1 = np.random.rand(3, 1)
b1 = np.random.rand(1, )

print("W1", W1)
print("b1", b1)


def numerical_derivative(f,x):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    delta_x = 1e-4

    while not it.finished:

        idx = it.multi_index
        temp_x = x[idx]

        x[idx] = float(temp_x) + delta_x
        fx1 = f(x)

        x[idx] = float(temp_x) - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = temp_x
        it.iternext()

    return grad


def loss_func(x, t ):
    y = np.dot(x, W1) + b1

    return (np.sum((t-y)**2))/(len(x))

def error_val(x, t ):
    y = np.dot(x, W1) + b1

    return (np.sum((t-y)**2))/(len(x))

def predict(x):
    return np.dot(x, W1) + b1


print("Initial error val", error_val(x_data,t_data))
print("Initial W1", W1)
print("Initial b1", b1)

for step in range(80001):

    f = lambda x: loss_func(x_data, t_data)
    learning_rate = 0.00001


    W1 -= learning_rate*numerical_derivative(f, W1)
    b1 -= learning_rate*numerical_derivative(f, b1)

    if(step % 400 == 0):
        print("step", step, "error_val", loss_func(x_data, t_data))

print(predict(np.array([73,80,75])))
