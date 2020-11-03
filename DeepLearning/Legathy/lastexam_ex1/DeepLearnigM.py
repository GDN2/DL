import numpy as np
import numerical_derivative_final as nd

loaded_data = np.loadtxt('./Picoscope(Second Test)v2.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:,2:4].reshape(240,2)
t_data = loaded_data[:,4].reshape(240,1)

input_nodes = 2
hidden_nodes = 6
output_nodes = 1

W2 = np.random.rand(input_nodes,hidden_nodes)
b2 = np.random.rand(hidden_nodes)

W3 = np.random.rand(hidden_nodes,output_nodes)
b3 = np.random.rand(output_nodes)
print(x_data,"\n",t_data)
print("W2 =", W2, ", W2.shape = ", W2.shape, ", b2 =", b2, ", b2.shape =", b2.shape)
print("W3 =", W3, ", W3.shape = ", W3.shape, ", b3 =", b3, ", b3.shape =", b3.shape)
learning_rate = 1e-6

def sigmoid(x):
    return 1/(1+np.exp(-x))

def feed_forward(xdata, tdata):
    delta = 1e-7
    z2 = np.dot(xdata, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2,W3) + b3
    y = a3 = sigmoid(z3)

    return -np.sum(tdata*np.log(y+delta)+(1-tdata)*np.log((1-y)+delta))


def loss_val(xdata, tdata):
    delta = 1e-7
    z2 = np.dot(xdata, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    y = a3 = sigmoid(z3)

    return -np.sum(tdata * np.log(y + delta) + (1 - tdata) * np.log((1 - y) + delta))

def predict(x_data):
    z2 = np.dot(x_data, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2,W3) + b3
    y = a3 = sigmoid(z3)

    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result;
f = lambda x : feed_forward(x_data, t_data)

print("Initial loss value = ", loss_val(x_data, t_data))

for step in range(1201):
    W2 -= learning_rate + nd.numerical_derivative(f, W2)
    b2 -= learning_rate + nd.numerical_derivative(f, b2)
    W3 -= learning_rate + nd.numerical_derivative(f, W3)
    b3 -= learning_rate + nd.numerical_derivative(f, b3)

    if (step % 400 == 0):
        print("step =",step,", loss value =", loss_val(x_data, t_data))

for data in x_data:
    (real_val, logical_val) = predict(data)
    print(data, "real_val", real_val, "logical_val", logical_val)