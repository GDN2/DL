import numpy as np

class LogisticRegression:
    def __init__(self, x_data, t_data, input_nodes, hidden_nodes, output_nodes, learning_rate = 1e-2, steps = 8001, delta_x=1e-5):
        self.x_data = x_data.reshape(-1,input_nodes)
        self.t_data = t_data.reshape(-1,output_nodes)
        self.W2 = np.random.rand(input_nodes,hidden_nodes)
        self.b2 = np.random.rand(hidden_nodes)
        self.W3 = np.random.rand(hidden_nodes, output_nodes)
        self.b3 = np.random.rand(output_nodes)
        self.learning_rate = learning_rate
        self.delta_x = delta_x
        self.steps = steps

    def sigmoid(self, z):
        return (1/(1+np.exp(-z))) # exp에는 delta_x필요 없음

    def loss_func(self):
        z2 = np.dot(self.x_data, self.W2) + self.b2 # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2, self.W3) + self.b3 # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        y = a3 = self.sigmoid(z3)
        E = -np.sum(self.t_data * np.log(y + self.delta_x) + (1 - self.t_data) * np.log(1 - y + self.delta_x))
        return E

    def error_val(self):
        z2 = np.dot(self.x_data, self.W2) + self.b2
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        y = a3 = self.sigmoid(z3)

        E = -np.sum(self.t_data * np.log(y + self.delta_x) + (1 - self.t_data) * np.log(1 - y + self.delta_x))
        return E
    
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

    def train(self):
        f = lambda x : self.loss_func()
        print( "Initial W2 =", self.W2, ", \nb2 =", self.b2, ",\nW3 =", self.W3, ", \nb3 =", self.b3,
               "\nInitial error value =", self.error_val())
        for step in range(self.steps):

            self.W2 -= self.learning_rate * self.numerical_derivative(f, self.W2)
            self.b2 -= self.learning_rate * self.numerical_derivative(f, self.b2)
            self.W3 -= self.learning_rate * self.numerical_derivative(f, self.W3)
            self.b3 -= self.learning_rate * self.numerical_derivative(f, self.b3)

            if (step % 400 == 0):
                print("step =", step, "error_value =", self.error_val())
                '''print("step =", step, "error_value =", self.error_val(), "W2 =", self.W2, ",W3 =", self.W3,
              ", b2 =", self.b2, ", b3 =", self.b3, "W3 =", self.W3, ", b3 =", self.b3)'''

    def predict(self,x):
        z2 = np.dot(x, self.W2) + self.b2
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        y = a3 = self.sigmoid(z3)
        result = 0
        if y >= 0.5:
            result = 1
            return y, result # return indent위치 주의하고 마지막에 한 번 더 return 하지 말기
        elif y < 0.5:
            result = 0
            return y, result

    def accuracy(self,x_data,t_data):
        matched_list = []
        notmatched_list = []
        accuracy_rate = 0

        for index in range(len(x_data)):
            (real_val, logical_val) = self.predict(x_data[index])
            if logical_val == t_data[index]:
                accuracy_rate += 1
                matched_list.append(index)
            else:
                notmatched_list.append(index)
        accuracy_rate = accuracy_rate/len(x_data)

        return accuracy_rate, matched_list, notmatched_list

    def accuracy_v2(self,test_data):
        matched_list = []
        notmatched_list = []
        accuracy_rate = 0
        x_data = test_data[:,:-1]
        t_data = test_data[:,[-1]]
        for index in range(len(x_data)):
            (real_val, logical_val) = self.predict(x_data[index])

            if logical_val == t_data[index]:
                matched_list.append(index)
                accuracy_rate += 1
            else:
                notmatched_list.append(index)
        accuracy_rate = accuracy_rate/len(x_data)
        return accuracy_rate, matched_list, notmatched_list

# 실수로 값 받기)
x_data = np.array([0,0,0,1,1,0,1,1]).reshape(-1,2)
and_t_data = np.array([0,0,0,1]).reshape(-1,1)
or_t_data = np.array([0,1,1,1]).reshape(-1,1)
nand_t_data = np.array([1,1,1,0]).reshape(-1,1)
xor_t_data = np.array([0,1,1,0]).reshape(-1,1)
and_test_data = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
xor_test_data = np.array([[0,0,0],[1,0,1],[0,1,1],[1,1,0]])
or_test_data = np.array([[0,0,0],[1,0,1],[0,1,1],[1,1,1]])
training_data = np.array([[0,0,0],[1,0,1],[0,1,1],[1,1,0]])
training_data = training_data.reshape(-1,3)
traing_data_x = training_data[:,:-1]
traing_data_t = training_data[:,[-1]]
input_data = np.array([[1,2,3,4],[0,1,2,0],[0,0,1,1]]).reshape(-1,4)
target_data = np.array([[0,1,0]]).reshape(-1,1)
print(input_data)
print(target_data)

obj = LogisticRegression(x_data,xor_t_data,2,2,1,1e-2,16001)
obj.train()

accuracy_rat, matched_list,notmatched_list = obj.accuracy(x_data,xor_t_data)
print(matched_list)
print(notmatched_list)
print(accuracy_rat)

accuracy_rat2, matched_list2,notmatched_list2 = obj.accuracy_v2(xor_test_data)
print(matched_list2)
print(notmatched_list2)
print(accuracy_rat2)
real_val, logical_val = obj.predict(np.array([[1,0]]))
print(real_val, logical_val)
