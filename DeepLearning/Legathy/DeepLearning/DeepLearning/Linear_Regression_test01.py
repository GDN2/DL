import numpy as np
import datetime

class LinearRegressionTest:
    def __init__(self, x_data, t_data, input_nodes, output_nodes, learning_rate = 1e-2,steps = 8001, delta_x=1e-4):
        self.x_data = x_data
        self.t_data = t_data
        self.__W = np.random.rand(input_nodes,output_nodes)
        self.__b = np.random.rand(output_nodes)
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
        y = np.dot(self.x_data, self.__W) + self.__b

        return (np.sum((self.t_data-y)**2))/len(self.x_data)

    def error_val(self):
        y = np.dot(self.x_data, self.__W) + self.__b

        return (np.sum((self.t_data-y)**2))/len(self.x_data)

    def predict(self, x):
        y = np.dot(x,self.__W) + self.__b

        return y

    def train(self):
        start_time = datetime.datetime.now()
        f = lambda x : self.loss_func()
        print("Initial error value =", self.error_val(), "Initial W =", self.__W, ", b =", self.__b)

        for step in range(self.steps):

            self.__W -= self.learning_rate * self.numerical_derivative(f, self.__W)
            self.__b -= self.learning_rate * self.numerical_derivative(f, self.__b)

            if (step % 400 == 0):
                print("step =", step, "error_value =", self.error_val(), "W =", self.__W, " b =", self.__b)
        end_time = datetime.datetime.now()
        print("소요시간", end_time - start_time)

    def getW(self):
        return self.__W

    def getb(self):
        return self.__b
# 실수로 값 받기

loaded_column = 4
x_column = 3
t_column = 1
input_nodes = 3 #x_column, 즉, 들어오는 데이터의 열의 수(input_data의 nodes 수)
output_nodes = 1 #t_column, 즉, 나가는 데이터의 열의 수 (output_data의 nodes 수)
learning_rate = 1e-7 # for data-01.csv 5.103e-5
steps = 4001


#x_data = np.array([x for x in range(-1000,2000)]).reshape(-1,1)
#t_data = np.array([2*t-1 for t in range(-1000,2000)]).reshape(-1,1)
loaded_data = np.loadtxt('./data/data-01.csv', delimiter=',', dtype=np.float32).reshape(-1,loaded_column)
print(loaded_data)
x_data = loaded_data[:,0:x_column].reshape(-1, x_column)
print(x_data)
t_data = loaded_data[:, -1].reshape(-1, t_column)
print(t_data)
obj = LinearRegressionTest(x_data, t_data, input_nodes, output_nodes, learning_rate,steps)
print("W =", obj.getW(), ",W.shape =", obj.getW().shape, ", b =", obj.getb(), ", b.shape =", obj.getb().shape)
obj.train()

for data in x_data: #전체 예측
    print(obj.predict(data))
print(obj.predict(np.array([50,50,50])))





