import numpy as np
import numerical_derivative_final as nd

loaded_data = np.loadtxt('./regression_testdata_05.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]

W = np.random.rand(4,1)
b = np.random.rand(1)
print("x_data =", x_data, "t_data =", t_data)
print("W =", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

def loss_func(x,t):
    y = np.dot(x,W) + b

    return (np.sum((t-y)**2))/(len(x))


def error_val(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y) ** 2)) / (len(x))

f = lambda x : loss_func(x_data, t_data)

def predict(x):
    y = np.dot(x,W) + b
    return y

learning_rate = 1e-3
print("Initial error value =", error_val(x_data, t_data), "Initial W =", W, "\n", ", b =", b)

for step in range(8001):
    W -= learning_rate * nd.numerical_derivative(f,W)
    b -= learning_rate * nd.numerical_derivative(f,b)
    if(step % 400 ==0):
        print("step =", step, "error value =", error_val(x_data,t_data), "W =", W, ", b =",b)

x = np.array([3,5,6,2]).reshape(1,4)
print(predict(x))