
import numpy as np

class Linear_regression:
    def __init__(self, x_data, t_data, input_nodes, hidden_nodes, output_nodes, learning_rate = 1e-2, steps = 8001, delta_x=1e-4):
        self.x_data = x_data
        self.t_data = t_data
        self.W1 = np.random.rand(input_nodes,hidden_nodes)
        self.b1 = np.random.rand(hidden_nodes)
        self.W2 = np.random.rand(hidden_nodes, output_nodes)
        self.b2 = np.random.rand(output_nodes)
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
        y1 = np.dot(self.x_data, self.W1) + self.b1

        y2 = np.dot(y1, self.W2) + self.b2

        return (np.sum((self.t_data-y2)**2))/len(self.x_data)

    def error_val(self):
        y1 = np.dot(self.x_data, self.W1) + self.b1

        y2 = np.dot(y1, self.W2) + self.b2

        return (np.sum((self.t_data - y2) ** 2)) / len(self.x_data)
    def predict(self, x):
        y1 = np.dot(x, self.W1) + self.b1

        y2 = np.dot(y1, self.W2) + self.b2

        return y2

    def linear_regression(self):
        f = lambda x : self.loss_func()
        print("Initial error value =", self.error_val(), "Initial W1 =", self.W1, ",W2 =", self.W2,
              ", b1 =", self.b1, ", b2 =", self.b2, "W2 =", self.W2, ", b2 =", self.b2)

        for step in range(self.steps):

            self.W1 -= self.learning_rate * self.numerical_derivative(f, self.W1)
            self.b1 -= self.learning_rate * self.numerical_derivative(f, self.b1)
            self.W2 -= self.learning_rate * self.numerical_derivative(f, self.W2)
            self.b2 -= self.learning_rate * self.numerical_derivative(f, self.b2)

            if (step % 400 == 0):
                print("step =", step, "error_value =", self.error_val(), "W1 =", self.W1, ",W2 =", self.W2,
              ", b1 =", self.b1, ", b2 =", self.b2, "W2 =", self.W2, ", b2 =", self.b2)


# 실수로 값 받기

#x_data = np.array([1,2,3,4,5,6]).reshape(3,2)
#y_data = np.array([19,41,63]).reshape(3,1)
loaded_data = np.loadtxt('./data/data-01.csv', delimiter=',', dtype=np.float32).reshape(-1,4)
print(loaded_data)
x_data = loaded_data[:,0:3].reshape(-1,3)
print(x_data)
t_data = loaded_data[:, -1].reshape(-1,1)
print(t_data)
obj = Linear_regression(x_data, t_data, 3, 3, 1, 0.00003, 8001)
print("W1 =", obj.W1, ",W2 =", obj.W2, ", b1 =", obj.b1, ", b2 =", obj.b2)
obj.linear_regression()
test_val = np.array([100,98,81])
predict_val = obj.predict(test_val)

print(obj.predict(np.array([100,98,81])))
#print(obj.predict(x_data))
#print(predict_val)





