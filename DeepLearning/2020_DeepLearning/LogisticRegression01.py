import numpy as np

loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/AND.txt', delimiter=',', dtype=np.float32)
x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]

W1 = np.random.rand(2,1)
b1 = np.random.rand(1)

learning_rate = 1e-2
input_nodes = x_data.shape[1]
output_nodes = t_data.shape[1]

steps = 8001

print("x_data", x_data)
print("t_data", t_data)
print("load_data", loaded_data)

def numerical_derivative(f,x):
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

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def loss_func(x,t):
    delta_x = 1e-4
    Z2 = np.dot(x,W1) + b1
    A2 = sigmoid(Z2)
    y = A2

    return -(np.sum(t_data * np.log(y + delta_x) + (1-t_data) * np.log(1-y+delta_x )))


def predict(x):
    Z2 = np.dot(x,W1) + b1
    A2 = sigmoid(Z2)
    y = A2

    if y>0.5:
        return 1

    if y<0.5:
        return 0

print("Initial loss_val", loss_func(x_data, t_data))
print("Initial W1", W1)
print("Initial b1", b1)

for step in range(steps):
    f = lambda W : loss_func(x_data, t_data)

    W1 -= learning_rate * numerical_derivative(f, W1)
    b1 -= learning_rate * numerical_derivative(f, b1)

    if (step % 400 == 0):
        print("step", step, "loss_val", loss_func(x_data, t_data))


for i in range(len(x_data)):
    print("predicted", predict(x_data[i]), "real_val", t_data[i])