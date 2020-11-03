import numpy as np

class LinearRegression:

    def __init__(self, x_data, t_data, hidden_nodes, learning_rate, steps):

        self.x_data = x_data
        self.t_data = t_data

        self.input_nodes = x_data.shape[1]
        self.output_nodes = t_data.shape[1]
        self.hidden_nodes = hidden_nodes

        self.learning_rate = learning_rate
        self.steps = steps


        self.W1 = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.b1 = np.random.rand(self.hidden_nodes)

        self.W2 = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.b2 = np.random.rand(self.output_nodes)

    def numerical_derivative(self, f, x):
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

            grad[idx] = (fx1 - fx2) / (2 * delta_x)

            x[idx] = temp_val
            it.iternext()

        return grad

    def loss_func(self):
        A2 = np.dot(self.x_data, self.W1) + self.b1
        A3 = np.dot(A2, self.W2) + self.b2
        y = A3

        return np.sum((t_data - y) ** 2) / len(self.x_data)

    def predict(self,x):
        A2 = np.dot(x, self.W1) + self.b1
        A3 = np.dot(A2, self.W2) + self.b2
        y = A3

        return y

    def run(self):

        print("Initial error_val", self.loss_func())
        print("Initial W1", self.W1)
        print("Initial b1", self.b1)
        print("Initial W2", self.W2)
        print("Initial b2", self.b2)

        for step in range(self.steps):
            f = lambda W: self.loss_func()

            self.W1 -= self.learning_rate * self.numerical_derivative(f, self.W1)
            self.b1 -= self.learning_rate * self.numerical_derivative(f, self.b1)

            self.W2 -= self.learning_rate * self.numerical_derivative(f, self.W2)
            self.b2 -= self.learning_rate * self.numerical_derivative(f, self.b2)

            if (step % 400 == 0):
                print("step", step, "error_val", self.loss_func())


loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/airplane_test_norm.csv', delimiter=',', dtype=np.float32)
x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]

linear = LinearRegression(x_data, t_data, 4, 1e-4, 8001)

linear.run()

for i in range(len(x_data)):
    print("predicted_val", linear.predict(x_data[i]), "real_val", t_data[i])

