import numpy as np
import numerical_derivative_final as nd


class MyDeep:

    def __init__(self,xdata,tdata,input_nodes,hidden_nodes,output_nodes):
        self.xdata = xdata
        self.tdata = tdata

        self.W2 = np.random.rand(input_nodes, hidden_nodes)
        self.b2 = np.random.rand(hidden_nodes)
        self.W3 = np.random.rand(hidden_nodes,output_nodes)
        self.b3 = np.random.rand(output_nodes)
        self.delta = 1e-7

    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))

    def feed_forward(self):
        z2 = np.dot(self.xdata, self.W2)+self.b2
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2,self.W3) + self.b3
        y = a3 = self.sigmoid(z3)

        return -np.sum(self.tdata*np.log(y+self.delta)+(1-self.tdata)*np.log(1-y+self.delta))

    def predict(self,xdata):
        z2 = np.dot(xdata, self.W2) + self.b2
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        y = a3 = self.sigmoid(z3)

        if (y > 0.5):
            result =  1
        if (y < 0.5):
            result = 0
        return y, result

xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_tdata = np.array([0, 1, 1, 0]).reshape(4, 1)
obj1 = MyDeep(xdata,xor_tdata,2,6,1)
learning_rate = 1e-4
f = lambda x : obj1.feed_forward()

for step in range(10001):
    obj1.W2 -= learning_rate*nd.numerical_derivative(f, obj1.W2)
    obj1.b2 -= learning_rate*nd.numerical_derivative(f,obj1.b2)
    obj1.W3 -= learning_rate*nd.numerical_derivative(f,obj1.W3)
    obj1.b3 -= learning_rate*nd.numerical_derivative(f, obj1.b3)

    if( step % 400 ==0):
        print("step =", step, "loss val =", obj1.feed_forward(), "obj1.W2", obj1.W2, "obj1.b2", obj1.b2, "obj1.W3", obj1.W3, "obj1.b3", obj1.b3)

for data in xdata:
    (real_val, logical_val) = obj1.predict(data)
    print("real_val", real_val, "logical_val", logical_val)