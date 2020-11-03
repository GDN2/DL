import numpy as np
import numerical_derivative_final as nd
from sqlalchemy.orm.strategy_options import loader_option

x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(10,1)
t_data = np.array([0,0,0,0,0,0,1,1,1,1]).reshape(10,1)

W = np.random.rand(1,1)
b = np.random.rand(1)
print("W =", W, ", W.shape = ", W.shape, ", b =", b, ", b.shape =", b.shape)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss_func(x,t):

    delta = 1e-7

    z = np.dot(x,W) + b
    y = sigmoid(z)

    return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))


def error_val(x, t):
    delta = 1e-7

    z = np.dot(x, W) + b
    y = sigmoid(z)

    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) +delta))

def predict(x):

    z = np.dot(x,W) + b
    y = sigmoid(z)

    if y >= 0.5:
        result = 1
    if y < 0.5:
        result = 0

    return y,result

learning_rate = 1e-2

f = lambda x : loss_func(x_data, t_data)

print("Initial error value = ", error_val(x_data, t_data), "Initial W = ", W, "\n", ", b = ", b)

for step in range(10001):

    W -= learning_rate * nd.numerical_derivative(f, W)
    b -= learning_rate * nd.numerical_derivative(f, b)

    if(step % 400 == 0):
        print("step =", step, "error_value =", error_val(x_data, t_data), "W =", W, ", b=", b)

for x in range(1300):
    (real_val, logical_val) = predict(x/100)
    print(x/100, real_val, logical_val)
for x in range(21):
    (real_val, logical_val) = predict(x)
    print(x, real_val, logical_val)
