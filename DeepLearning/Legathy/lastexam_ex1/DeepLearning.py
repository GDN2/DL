import numpy as np
import numerical_derivative_final as nd

xdata = np.array([ [0,0],[0,1],[1,0],[1,1] ])
and_tdata = np.array([0,0,0,1]).reshape(4,1)
or_tdata = np.array([0,1,1,1]).reshape(4,1)
nand_tdata = np.array([1,1,1,0]).reshape(4,1)
xor_tdata = np.array([0,1,1,0]).reshape(4,1)

test_data = np.array([[0,0],[0,1],[1,0],[1,1]])

input_nodes = 2
hidden_nodes = 6
output_nodes = 1

W2 = np.random.rand(input_nodes,hidden_nodes)
b2 = np.random.rand(hidden_nodes)

W3 = np.random.rand(hidden_nodes,output_nodes)
b3 = np.random.rand(output_nodes)

learning_rate = 1e-2

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

def predict(xdata):
    z2 = np.dot(xdata, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2,W3) + b3
    y = a3 = sigmoid(z3)

    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result
f = lambda x : feed_forward(xdata, xor_tdata)

print("Initial loss value = ", loss_val(xdata, xor_tdata))

for step in range(8001):
    W2 -= learning_rate + nd.numerical_derivative(f, W2)
    b2 -= learning_rate + nd.numerical_derivative(f, b2)
    W3 -= learning_rate + nd.numerical_derivative(f, W3)
    b3 -= learning_rate + nd.numerical_derivative(f, b3)

    if (step % 400 == 0):
        print("step =",step,", loss value =", loss_val(xdata, xor_tdata))

for data in test_data:
    (real_val, logical_val) = predict(data)
    print("real_val", real_val, "logical_val", logical_val)