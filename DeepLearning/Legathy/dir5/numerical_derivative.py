import numpy as np

def numerical_derivative(f,x):
    delta_x = 1e-4
    return (f(x+delta_x)-f(x-delta_x))/(2*delta_x)
f = lambda x : x**2
print(numerical_derivative(f,3))

f = lambda x : 3*x*np.exp(x)

print(numerical_derivative(f,2))

