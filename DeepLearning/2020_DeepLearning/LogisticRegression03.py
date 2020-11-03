import numpy as np
from datetime import datetime


class LogisticRegression:

    def __init__(self, x_data, t_data, hidden_nodes, learning_rate, steps):

        self.x_data = x_data
        self.t_data = t_data

        self.input_nodes = self.x_data.shape[1]
        self.output_nodes = self.t_data.shape[1]
        self.hidden_nodes = hidden_nodes

        self.W2 = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.b2 = np.random.rand(self.hidden_nodes)

        self.W3 = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.b3 = np.random.rand(self.output_nodes)

        self.learning_rate = learning_rate
        self.steps = steps

    def numerical_dreivative(self, f, x):
        grad = np.zeros_like(x)
        delta_x = 1e-4
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            temp_val = x[idx]

            x[idx] = float(temp_val) + delta_x
            fx1 = f(x)

            x[idx] = float(temp_val) - delta_x
            fx2 = f(x)

            grad[idx] = (fx1 - fx2) / (2*delta_x)
            x[idx] = temp_val
            it.iternext()

        return grad

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def loss_func(self):
        delta_x = 1e-4
        Z2 = np.dot(self.x_data, self.W2) + self.b2
        A2 = self.sigmoid(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = self.sigmoid((Z3))

        y = A3

        return -(np.sum(self.t_data*np.log(y+delta_x) + (1-self.t_data)*np.log(1-y+delta_x)))

    def predict(self, x):
        Z2 = np.dot(x, self.W2) + self.b2
        A2 = self.sigmoid(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = self.sigmoid((Z3))

        y = A3

        if y > 0.5:
            return 1
        else:
            return 0

    def run(self):

        print("Initial W2", self.W2)
        print("Initial b2", self.b2)
        print("Initial W3", self.W3)
        print("Initial b3", self.b3)
        print("Initial loss_val", self.loss_func())
        print("x_data.shape", self.x_data.shape)
        print("t_data.shape", self.t_data.shape)

        start_time = datetime.now()

        for step in range(self.steps):

            f = lambda W : self.loss_func()

            self.W2 -= self.learning_rate * self.numerical_dreivative(f, self.W2)
            self.b2 -= self.learning_rate * self.numerical_dreivative(f, self.b2)

            self.W3 -= self.learning_rate * self.numerical_dreivative(f, self.W3)
            self.b3 -= self.learning_rate * self.numerical_dreivative(f, self.b3)


            if (step % 10 == 0):
                print("step", step, "loss_val", self.loss_func())

        end_time = datetime.now()
        print("Time", end_time - start_time)


loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/ThoracicSurgery_norm_test.csv', delimiter=',', dtype=np.float32)
x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]

logist = LogisticRegression(x_data, t_data, 8, 1e-2, 500)
logist.run()
count = 0
for i in range(len(x_data)):

    predicted_val = logist.predict(x_data[i])
    real_val = t_data[i]

    #print("predicted_val", predicted_val, "real_val", real_val)

    if (int(predicted_val) == (round(float(real_val)))):
        count = count + 1

print("Prediction rate", float(count/len(t_data)))
