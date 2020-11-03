import numpy as np
from datetime import datetime

class backpropagation:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.intput_data = np.zeros([1,input_nodes]) # input_data도 이런 방식으로 초기화해두기
        self.target_data = np.zeros([1,output_nodes])

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.W3 = np.random.randn(hidden_nodes, output_nodes) / np.sqrt(hidden_nodes/2)
        self.b3 = np.random.randn(output_nodes)

        self.W2 = np.random.randn(input_nodes, hidden_nodes) / np.sqrt(hidden_nodes/2)
        self.b2 = np.random.randn(hidden_nodes)

        self.Z1 = np.zeros([1, input_nodes])
        self.A1 = np.zeros([1, input_nodes])

        self.Z2 = np.zeros([1, hidden_nodes])
        self.A2 = np.zeros([1, hidden_nodes])

        self.Z3 = np.zeros([1, output_nodes])
        self.A3 = np.zeros([1, output_nodes])

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def numerical_derivative(self, f, x):
        grad = np.zeros_like(x)
        delta_x = 1e-4
        it = np.nditer(x, flag=['multi_index'], op_flag=['readwrite'])

        while not it.finished:

            idx = it.multi_index
            tmp_val = x[idx]

            x[idx] = float(tmp_val) + delta_x
            fx1 = f(x)

            x[idx] = float(tmp_val) - delta_x
            fx2 = f(x)

            grad[idx] = (fx1 - fx2) / (2*delta_x)
            x[idx] = tmp_val
            it.iternext()

        return grad

    def loss_func(self):

        delta_x = 1e-4
        self.Z1 = self.input_data
        self.A1 = self.Z1

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        y = self.A3

        return -(np.sum(self.target_data*np.log(y+delta_x) + (1-self.target_data)*np.log(1-y+delta_x)))

    def train(self, x_data, t_data):

        self.input_data = x_data
        self.target_data = t_data
        self.loss_func()

        loss3 = (self.A3 - self.target_data) * self.A3 * (1-self.A3)
        loss2 = np.dot(loss3, self.W3.T) * self.A2 * (1-self.A2)

        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss3)
        self.b3 = self.b3 -  self.learning_rate * loss3 # -=표시 말고 직접해주어야 브로드캐스트 됨

        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss2)
        self.b2 = self.b2 - self.learning_rate * loss2

    def predict(self, input_data):
        delta_x = 1e-4
        self.Z1 = self.input_data
        self.A1 = self.Z1

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        y = self.A3

        if y > 0.5:
            return 1
        elif y <0.5:
            return 0



loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/diabetes_trainV2_normV2.csv', delimiter=',', dtype=np.float32)
x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]
input_nodes = x_data.shape[1]
hidden_nodes = 8
output_nodes = t_data.shape[1]
learning_rate = 1
epochs = 20
steps = len(x_data)
print(output_nodes)

obj = backpropagation(input_nodes, hidden_nodes, output_nodes, learning_rate)

print("Initial W3.shape", obj.W3.shape)
print("Initial b3.shape", obj.b3.shape)
print("Initial W2.shape", obj.W2.shape)
print("Initial b2.shape", obj.b2.shape)
print("Initial A1.shape", obj.A1.shape)
print("Initial A2.shape", obj.A2.shape)
print("Initial A3.shape", obj.A3.shape)

start = datetime.now()
'''


'''
for epoch in range(epochs):
    for index in range(len(x_data)):
        input_data = x_data[index].reshape(1,-1) # 2차원으로 보내주기
        target_data = t_data[index].reshape(1,-1)
        obj.train(input_data, target_data)

        if (index % 100 == 0):
            print("step", index, "loss_val", obj.loss_func())
    if (epoch % 5 == 0):
        print("epoch", epoch, "loss_val", obj.loss_func())


end = datetime.now()
print("Time", end-start)


loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/diabetes_testV2_normV2.csv', delimiter=',', dtype=np.float32)
x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]


count = 0
for i in range(len(x_data)):
    y = obj.predict(x_data[i])
    if (round(float(y)) == round(float(t_data[i]))):
        count += 1
    if count < 30:
        print(int(round(float(y))), int(round(float(t_data[i]))))
print("prediction ratio", float(count/len(x_data)))