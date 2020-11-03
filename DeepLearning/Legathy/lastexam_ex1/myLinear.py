import numpy as np
import numerical_derivative_final as nd

class myLinear:
    def __init__(self,xdata,tdata,input_nodes,output_nodes):
        self.xdata = xdata
        self.tdata = tdata
        self.W = np.random.rand(input_nodes,output_nodes)
        self.b = np.random.rand(output_nodes)

    def loss_func(self):
        y = np.dot(self.xdata,self.W) + self.b
        return np.sum((self.tdata-y)**2)/len(self.xdata)

    def predict(self,xdata):
        y = np.dot(xdata,self.W) + self.b
        return y

loaded_data = np.loadtxt('./regression_testdata_05.csv', delimiter = ',', dtype=np.float32)
xdata = loaded_data[:,0:-1].reshape(-1,4)
tdata = loaded_data[:,-1].reshape(-1,1)
learning_rate = 1e-3
obj1 = myLinear(xdata,tdata,4,1)
f = lambda x : obj1.loss_func()


for step in range(1000001):
    if(step % 400 == 0):
        obj1.W -= learning_rate*nd.numerical_derivative(f,obj1.W)
        obj1.b -= learning_rate*nd.numerical_derivative(f,obj1.b)
        print("step", step, "error value =", obj1.loss_func(), "W =", obj1.W, "b =", obj1.b)

x = np.array([3,5,6,2]).reshape(1,4)
real_val = obj1.predict(x)
print(real_val)


