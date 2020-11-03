import numpy as np
import datetime

class Linear_regression:
    def __init__(self, x_data, t_data, input_nodes, output_nodes, learning_rate = 1e-2,steps = 8001, delta_x=1e-4):
        self.x_data = x_data
        self.t_data = t_data
        self.W = np.random.rand(input_nodes,output_nodes)
        self.b = np.random.rand(output_nodes)
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

    def loss_func2(self,x,t):
        y = np.dot(x, self.W) + self.b

        return (np.sum((t-y)**2))/len(x)

    def error_val2(self,x,t):
        y = np.dot(x, self.W) + self.b

        return (np.sum((t-y)**2))/len(x)

    def predict(self, x):
        y = np.dot(x,self.W) + self.b

        return y

    def linear_regression(self):
        start_time = datetime.datetime.now()
   #     f = lambda x : self.loss_func()
        print("Initial error value =", self.error_val2(self.x_data.reshape(-1,1),self.t_data.reshape(-1,1)), "Initial W =", self.W, ", b =", self.b)

        for step in range(self.steps):

            for index in range(len(self.x_data)):

                x = self.x_data[index]
                t = self.t_data[index]
                f = lambda x : self.loss_func2(x,t)
                self.W -= self.learning_rate * self.numerical_derivative(f, self.W)
                self.b -= self.learning_rate * self.numerical_derivative(f, self.b)

            if (step % 400 == 0):
                print("step =", step, "error_value =", self.error_val2(self.x_data.reshape(-1,1),self.t_data.reshape(-1,1)),
                      "W =", self.W, " b =", self.b)

        end_time = datetime.datetime.now()
        print("소요시간", end_time - start_time)
# 실수로 값 받기


loaded_column = 5
x_column = 1
t_column = 1
input_nodes = 1 #x_column, 즉, 들어오는 데이터의 열의 수(input_data의 nodes 수)
output_nodes = 1 #t_column, 즉, 나가는 데이터의 열의 수 (output_data의 nodes 수)
learning_rate = 1e-5 # for data-01.csv 5.103e-5
steps = 40001

x_data = np.array([x for x in range(-50,71)])
t_data = np.array([2*t-1 for t in range(-50,71)])
#loaded_data = np.loadtxt('./data/regression_testdata_03.csv', delimiter=',', dtype=np.float32).reshape(-1,loaded_column)
#print(loaded_data)
#x_data = loaded_data[:,0:x_column].reshape(-1, x_column)
#print(x_data)
#t_data = loaded_data[:, -1].reshape(-1, t_column)
#print(t_data)
obj = Linear_regression(x_data, t_data, input_nodes, output_nodes, learning_rate,steps)
print("W =", obj.W, ",W.shape =", obj.W.shape, ", b =", obj.b, ", b.shape =", obj.b.shape)
obj.linear_regression()

#for data in x_data: #전체 예측
#    print(obj.predict(data))
print(obj.predict(np.array([44])))





