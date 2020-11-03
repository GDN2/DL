import numpy as np
import matplotlib.pyplot as plt

W = np.random.rand(784,100).reshape(784*100,1)
W1 = np.random.randn(784,100).reshape(784*100,1)
W2 = np.sqrt(100)
print(W2)
W3 = np.sqrt(100/2)
W4 = (np.random.randn(784,100)/np.sqrt(100)).reshape(784*100,1)
W5 = (np.random.randn(784,100)/np.sqrt(100/2)).reshape(784*100,1)
W6 = (np.random.randn(784,100)/9).reshape(784*100,1)
a = np.array([0,1,2,3])
b = np.array([[0,1,2,3]])
A = []
for index in range(784*100):
    A.append(index)
A = np.array(A)
plt.plot(W6, A)
plt.show()
b1 = np.random.rand(1000,1)
b2 = np.random.randn(1000,1)
B = []
for index in range(1000):
    B.append(index)
#plt.plot(B, b2)
#plt.show()