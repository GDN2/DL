import numpy as np

def numerical_derivative(f,x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2)/(2*delta_x)

        x[idx] = tmp_val
        it.iternext()


    return grad

def func1(W):
    x = W[0]
    y = W[1]

    return (2*x + 3*x*y + np.power(y,3))

f = lambda W : func1(W)

W = np.array([1,2.0])

print(numerical_derivative(f,W))