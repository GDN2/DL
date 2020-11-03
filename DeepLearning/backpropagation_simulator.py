import numpy as np

W2 = np.array([1,1,2,1]).reshape(2,2)
W3 = np.array([2,1,1,1]).reshape(2,2)

T = np.array([2,1])

b2 = np.array([1,1])
b3 = np.array([0,2])

learning_rate = 0.1

A1 = np.array([1,2],ndmin=2)
A2 = np.random.rand(1,2)
A3 = np.random.rand(1,2)

def sigmoid(z):

    dic = { 0:-3,
            1:-2,
            2:-1,
            3:0,
            4:1,
            5:2,
            6:3,
            7:4
                 }
    z[0][0] = dic[z[0][0]]
    z[0][1] = dic[z[0][1]]

    return z

def feedforward():
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3)+b3
    A3 = sigmoid(Z3)
    return A2, A3

A2, A3 = feedforward()
print(A2)
print(A3)

def backpropagation(b3, W3, b2, W2):

    loss3 = (A3-T)*A3*(1-A3)
    loss2 = np.dot(loss3, W3.T)*A2*(1-A2)

    b3 = b3 - learning_rate*loss3
    W3 = W3 - learning_rate*np.dot(A2.T, loss3)

    b2 = b2 - learning_rate*loss2
    W2 = W2 - learning_rate*np.dot(A1.T, loss2)


    return b3, W3, b2, W2

b3, W3, b2, W2 = backpropagation(b3, W3, b2, W2)

print("backpropagation")
print("b3", b3)
print("W3", W3)
print("b2", b2)
print("W2", W2)