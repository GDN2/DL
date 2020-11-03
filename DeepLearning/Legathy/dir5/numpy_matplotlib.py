import numpy as np

A = np.array([1,2])

print(A, type(A))

from numpy import exp

result = exp(1)

print(result, type(result))

from numpy import *

result = exp(1) + log(1.7) + sqrt(2)
print(result, type(result))

A = [[1,0],[0,1]]
B = [[1,1],[1,1]]

print(A + B)

A = np.array([[1,0],[0,1]])
B = np.array([[1,1],[1,1]])

print(A+B)
print(type(A))
print(A.shape, B.shape)

A = np.array([1,2,3])
B = np.array([4,5,6])

print(A, B)
print(A.shape, B.shape)
print(A.ndim, B.ndim)
print(A+B)
print(A-B)
print(A*B)
print(A/B)

C = np.array([1, 2, 3])
print(C.shape)
print(C.ndim)
C = C.reshape(1, 3)
print(C.shape)
print(C.ndim)

A = np.array([[1,2,3],[4,5,6]])
B = np.array([[-1,-2],[-3,-4],[-5,-6]])
C = np.dot(A, B)
print(C)
D = 5
print(A+D)

A = np.array([[1,2],[3,4],[5,6]])
print(A)
print(A.T)
C = np.array([1,2,3,4,5])
D = C.T

E = C.reshape(1,5)
F = E.T
print(C)
print(D)
print(E)
print(F)

A = np.array([10,20,30,40,50,60]).reshape(3,2)



print(A)

print(A[0][0], A[1][1], A[2],"\n")

print(A[0:-1,1:2])
print(A[:,:].shape)
print(A[:,0].shape)
print(A[0:-1,1:2])


A = np.array([[10,20,30,40],[50,60,70,80]])
print(A)
print("A.shape ==", A.shape, "\n")

it = np.nditer(A, flags=['multi_index'], op_flags=['readwrite'])

while not it.finished:

    idx = it.multi_index
    print("current value -> ", A[idx])

    it.iternext()

with open("./data-01.csv", 'w') as f:
    f.write("73, 80, 75,15\n293, 88, 93, 185")

loaded_data = np.loadtxt('./data-01.csv', delimiter = ',', dtype=np.float32)
x_data = loaded_data[:,0:-1]
y_data = loaded_data[:,-1]
print("x_data.dim =", x_data.ndim, ", x_data.shape = ", x_data.shape)
print("y_data.dim =", y_data.ndim, ", y_data.shape = ", y_data.shape)
print(x_data)
print(y_data)

random1 = np.random.rand(3)
random2 = np.random.rand(1, 3)
random3 = np.random.rand(3, 1)

print(random1)
print(random2)
print(random3)

X = np.array([2,4,5,7,8])

print(np.sum(X))
print(np.exp(X))
print(np.log(X))

