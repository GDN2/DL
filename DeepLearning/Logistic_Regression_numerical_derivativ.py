import numpy as np
from datetime import datetime

class Logistic_Regression_nd:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.55):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.W2 = np.random.randn(input_nodes, hidden_nodes) / np.sqrt(input_nodes/2)
        self.b2 = np.random.rand(hidden_nodes)
        self.W3 = np.random.randn(hidden_nodes, output_nodes) / np.sqrt(hidden_nodes/2)
        self.b3 = np.random.rand(output_nodes)

        self.A3 = np.zeros([1,output_nodes])
        self.Z3 = np.zeros([1,output_nodes])
        self.A2 = np.zeros([1,hidden_nodes])
        self.Z2 = np.zeros([1,hidden_nodes])
        self.A1 = np.zeros([1,input_nodes])
        self.Z1 = np.zeros([1,input_nodes])

        self.input_data = np.zeros([1,input_nodes])
        self.target_data = np.zeros([1, output_nodes])

        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def loss_func(self):
        delta = 1e-7
        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        E = -np.sum(self.target_data*np.log(self.A3+delta)+(1-self.target_data)*np.log(1-self.A3+delta))

        return E

    def error_val(self):
        delta = 1e-7
        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        E = -np.sum(self.target_data * np.log(self.A3 + delta) + (1 - self.target_data) * np.log(1 - self.A3 + delta))

        return E

    def numerical_derivative(self,f,x):
        delta_x = 1e-7
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index

            tmp_val = x[idx]
            x[idx] = float(tmp_val)+delta_x
            f1 = f(x)

            x[idx] = float(tmp_val)-delta_x
            f2 = f(x)
            grad[idx] = (f1-f2)/2*delta_x

            x[idx] = tmp_val
            it.iternext()
        return grad

    def train(self, input_data, target_data):
        self.intput_data = input_data
        self.target_data = target_data
        f = lambda x : self.loss_func()

        self.b3 = self.b3 - self.learning_rate*self.numerical_derivative(f, self.b3)
        self.W3 = self.W3 - self.learning_rate*self.numerical_derivative(f, self.W3)
        self.b2 = self.b2 - self.learning_rate*self.numerical_derivative(f, self.b2)
        self.W2 = self.W2 - self.learning_rate*self.numerical_derivative(f, self.W2)

    def predict_onehot(self, input_data):

        self.Z2 = np.dot(input_data, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        return np.argmax(self.A3, axis=1)

    def accuracy_onehot(self, input_data, target_data):
        matched_list = []
        notmatched_list = []
        tmp_list = []
        index_label_list = []
        accuracy_rate = 0

        for index in range(len(target_data)):
            argmax = self.predict_onehot(np.array((input_data[index]/255+0.01) * 0.99,ndmin=2))
            if argmax == target_data[index]:
                accuracy_rate += 1
                matched_list.append(index)
            else:
                notmatched_list.append(index)
                tmp_list.append(index)
                tmp_list.append(target_data)
                tmp_list.append(argmax)
                index_label_list.append(tmp_list)
                tmp_list = []

        accuracy_rate = accuracy_rate / len(input_data)

        return accuracy_rate, matched_list, notmatched_list, index_label_list

loaded_txt = np.loadtxt('./data/mnist_train.csv', delimiter=',', dtype=np.float32)
train_x_data = loaded_txt[:,1:]
train_t_data = loaded_txt[:,[0]]
print("Data load is done")

obj = Logistic_Regression_nd(784, 1, 10)
start_time = datetime.now()
for epoch  in range(1):
    for index in range(10):
        input_data = (train_x_data/255 +0.01) * 0.99
        target_data = np.zeros(10) + 0.01
        target_data[int(train_t_data[index, 0])] = 0.99
        obj.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2))
        if index % 1 == 0:
            print("index =", index, "error_val =", obj.error_val())
end_time = datetime.now()
print("Time spend ==>", end_time - start_time)

loaded_txt2 = np.loadtxt('./data/mnist_test.csv', delimiter=',', dtype=np.float32)
test_x_data = loaded_txt2[:,1:]
test_t_data = loaded_txt2[:,[0]]
print("Data load is done")
accuracy_rat, matched_list, notmatched_list, index_label_list = obj.accuracy_onehot(test_x_data, test_t_data)

print(accuracy_rat)



