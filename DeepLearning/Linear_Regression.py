import numpy as np
import datetime

class Linear_regression:
    def __init__(self, x_data, t_data, input_nodes, output_nodes, learning_rate = 1e-2,steps = 8001, delta_x=1e-4):
        self.x_data = x_data
        self.t_data = t_data
        self.W = np.random.rand(input_nodes,output_nodes)
        #self.W = np.array([[0.35593829],[0.54250078],[1.16738458]])
        self.b = np.random.rand(output_nodes)
        #self.b = np.array([-4.32969629]) #error_value = 5.737807
        self.learning_rate = learning_rate
        self.delta_x = delta_x
        self.steps = steps

    def numerical_derivative(self,f,x):
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            
            tmp_val = x[idx]
            x[idx] = float(tmp_val)+self.delta_x
            f1 = f(x)
            x[idx] = float(tmp_val)-self.delta_x
            f2 = f(x)
           
            grad[idx] = (f1-f2)/(2*self.delta_x)# 분자에 괄호
            
            x[idx] = tmp_val
            it.iternext()
        
        return grad #return indent while에 맞추기

    def loss_func(self):
        y = np.dot(self.x_data, self.W) + self.b

        return (np.sum((self.t_data-y)**2))/len(self.x_data)

    def error_val(self):
        y = np.dot(self.x_data, self.W) + self.b

        return (np.sum((self.t_data-y)**2))/len(self.x_data)

    def predict(self, x):
        y = np.dot(x,self.W) + self.b

        return y

    def linear_regression(self):
        start_time = datetime.datetime.now()
        f = lambda x : self.loss_func()
        print("Initial error value =", self.error_val(), "Initial W =", self.W, ", b =", self.b)

        for step in range(self.steps):

            self.W -= self.learning_rate * self.numerical_derivative(f, self.W)
            self.b -= self.learning_rate * self.numerical_derivative(f, self.b)

            if (step % 400 == 0):
                print("step =", step, "error_value =", self.error_val(), "W =", self.W, " b =", self.b)
        end_time = datetime.datetime.now()
        print("소요시간", end_time - start_time)

# 실수로 값 받기

#x_data = np.array([1,2,3,4,5,6]).reshape(3,2)
#y_data = np.array([19,41,63]).reshape(3,1)
loaded_data = np.loadtxt('./data/data-01.csv', delimiter=',', dtype=np.float32).reshape(-1,4)
print(loaded_data)
x_data = loaded_data[:,0:3].reshape(-1,3)
print(x_data)
t_data = loaded_data[:, -1].reshape(-1,1)
print(t_data)
obj = Linear_regression(x_data, t_data, 3, 1, 0.00005103,80001)
print("W =", obj.W, ",W.shape =", obj.W.shape, ", b =", obj.b, ", b.shape =", obj.b.shape)
obj.linear_regression()
test_val = np.array([10,11,33])
predict_val = obj.predict(test_val)
for data in x_data:
    print(obj.predict(data))
print(predict_val)





