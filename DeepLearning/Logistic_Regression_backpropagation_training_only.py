import numpy as np
from datetime import datetime

class LogisticRegression:
    def __init__(self, train_data, input_nodes, hidden_nodes, output_nodes, learning_rate = 1e-2, steps = 8001, delta_x=1e-5):

        self.train_data = train_data
        self.x_data = train_data[:,:-1]
        self.t_data = train_data[:,[-1]]

        self.hidden_nodes = hidden_nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.W2 = np.random.rand(input_nodes,hidden_nodes)
        self.b2 = np.random.rand(hidden_nodes)
        self.W3 = np.random.rand(hidden_nodes, output_nodes)
        self.b3 = np.random.rand(output_nodes)

        self.learning_rate = learning_rate
        self.delta_x = delta_x
        self.steps = steps

        self.z2 = np.dot(self.x_data, self.W2) + self.b2 # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3 # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.a3 = self.sigmoid(self.z3)
        self.E = -np.sum(self.t_data * np.log(self.a3 + self.delta_x) + (1 - self.t_data) * np.log(1 - self.a3 + self.delta_x))

    def sigmoid(self, z):
        return (1/(1+np.exp(-z))) # exp에는 delta_x필요 없음

    def loss_func(self):
        self.z2 = np.dot(self.x_data, self.W2) + self.b2 # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3 # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.a3 = self.sigmoid(self.z3)
        self.E = -np.sum(self.t_data * np.log(self.a3 + self.delta_x) + (1 - self.t_data) * np.log(1 - self.a3 + self.delta_x))
        return self.E

    def error_val(self):
        self.z2 = np.dot(self.x_data, self.W2) + self.b2 # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3 # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.a3 = self.sigmoid(self.z3)
        self.E = -np.sum(self.t_data * np.log(self.a3 + self.delta_x) + (1 - self.t_data) * np.log(1 - self.a3 + self.delta_x))
        return self.E
    
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

    def backpropagation(self, Wb):

        if Wb == 'b2':
            one_matrix = np.ones(len(self.x_data))
            forb2 = np.dot((np.dot((self.a3 - self.t_data) * self.a3 * (1 - self.a3), self.W3.T) * self.a2 * (1 - self.a2)).T,one_matrix).T
            return forb2
        elif Wb == 'W2':
            forW2 = np.dot((np.dot((self.a3 - self.t_data) * self.a3 * (1 - self.a3), self.W3.T) * self.a2 * (1 - self.a2)).T,self.x_data).T
            return forW2
        elif Wb == 'b3':
            one_matrix = np.ones(len(self.x_data))
            forb3 = np.dot(((self.a3 - self.t_data) * self.a3 * (1 - self.a3)).T, one_matrix).T
            return forb3
        elif Wb == 'W3':
            forW3 = np.dot(((self.a3 - self.t_data) * self.a3 * (1 - self.a3)).T, self.a2).T
            return forW3
        else:
            print("No Type!")


    def train(self):
        start_time = datetime.now()
        f = lambda x : self.loss_func()
        print( "Initial W2 =", self.W2, ", \nb2 =", self.b2, ",\nW3 =", self.W3, ", \nb3 =", self.b3,
               "\nInitial error value =", self.error_val())
        for step in range(self.steps):

            self.b2 -= self.learning_rate * self.backpropagation('b2')
            self.W2 -= self.learning_rate * self.backpropagation('W2')
            self.b3 -= self.learning_rate * self.backpropagation('b3')
            self.W3 -= self.learning_rate * self.backpropagation('W3')

            if (step % 400 == 0):
                print("step =", step, "error_value =", self.error_val())
        end_time = datetime.now()
        print("Training Time =>", end_time - start_time)

    def predict(self,x):
        self.z2 = np.dot(x, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        result = 0

        if self.a3 > 0.5:
            result = 1
            return self.a3, result # return indent위치 주의하고 마지막에 한 번 더 return 하지 말기
        elif self.a3 < 0.5:
            result = 0
            return self.a3, result

    def predict_onehot(self,x):
        self.z2 = np.dot(x, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        result = 0

        return np.argmax(self.a3, axis=0)

    def accuracy(self,x_data,t_data):
        print(x_data)
        matched_list = []
        notmatched_list = []
        index_label_list = []
        temp_list = []
        accuracy_rate = 0
        count1 = 0
        count2 = 0

        for index in range(len(x_data)):
            (real_val, logical_val) = self.predict(x_data[index])
            if logical_val == int(np.rint(t_data[index])):
                print(logical_val, t_data[index])
                accuracy_rate += 1
                matched_list.append(index)
                if logical_val == 0:
                    count1 +=1

            else:
                notmatched_list.append(index)
                temp_list.append(index)
                temp_list.append(logical_val)
                temp_list.append(int(np.rint(t_data[index])))
                index_label_list.append(temp_list)
                temp_list = []
                if logical_val == 0:
                    count2 +=1

        accuracy_rate = accuracy_rate/len(x_data)

        print("matched_list ==>", matched_list)
        print("notmatched_list ==>", notmatched_list)
        print("index_lable_list ==>", index_label_list)
        print("accuracy_rate ==>", accuracy_rate)
        print(count1)
        print(count2)

        return accuracy_rate, matched_list, notmatched_list, index_label_list

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
        print(matched_list)
        print(notmatched_list)
        print(accuracy_rate)
        return matched_list, notmatched_list, accuracy_rate

    def accuracy_onehot(self,x_data, t_data):
        matched_list = []
        notmatched_list = []
        accuracy_rate = 0

        for index in range(len(x_data)):

            argmax = self.predict_onehot(x_data[index])
            if argmax == t_data[index]:
                accuracy_rate += 1
                matched_list.append(index)
            else:
                notmatched_list.append(index)

        accuracy_rate = accuracy_rate/len(x_data)

        print("matched_list =>", matched_list)
        print("notmatched_list =>", notmatched_list)
        print("accuracy_rate =>", accuracy_rate)

        return accuracy_rate, matched_list, notmatched_list

def make_onehot_t_data(t_data, type):
    size_matrix = len(t_data) * (type)  # type 개수 주의
    zero_matrix = np.zeros(size_matrix).reshape(len(t_data), (type))
    for index in range(len(t_data)):
        zero_matrix[index, t_data[index]] = 1
    return zero_matrix



# 실수로 값 받기)
x_data = np.array([0,0,0,1,1,0,1,1]).reshape(-1,2)
and_t_data = np.array([0,0,0,1]).reshape(-1,1)
or_t_data = np.array([0,1,1,1]).reshape(-1,1)
nand_t_data = np.array([1,1,1,0]).reshape(-1,1)
xor_t_data = np.array([0,1,1,0]).reshape(-1,1)
and_test_data = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
xor_test_data = np.array([[0,0,0],[1,0,1],[0,1,1],[1,1,0]])
or_test_data = np.array([[0,0,0],[1,0,1],[0,1,1],[1,1,1]])
xor_onehotencoding_data = np.array([[1,0],[0,1],[0,1],[1,0]])

data_name = 'ThoracicSurgery'

loaded_data = np.loadtxt('./data/wine_norm_train.csv' , delimiter = ',', dtype = np.float32)

train_loaded_data = np.loadtxt('./data/wine_norm_train.csv' , delimiter = ',', dtype = np.float32)
train_data = train_loaded_data[:, :-1]
train_t_data = train_loaded_data[:, [-1]]

test_loaded_data = np.loadtxt('./data/wine_norm_train.csv' , delimiter = ',', dtype = np.float32)
test_data = test_loaded_data[:, :-1]
test_t_data = test_loaded_data[:, [-1]]

obj = LogisticRegression(loaded_data, loaded_data.shape[1]-1, 6, 1, 1e-5, 20001)
obj.train()
accuracy_rat, matched_list, notmatched_list, index_label_list = obj.accuracy(test_data, test_t_data)
print(len(test_data))

'''loaded_data = np.loadtxt('./data/mnist_train.csv', delimiter = ',', dtype = np.float32)
training_data_mnist = loaded_data[:, 1:]
training_t_data'''


'''predict_onehot_val = obj.predict_onehot(np.array([[0,0]]))
predict_onehot_val = obj.predict_onehot(np.array([[1,0]]))
predict_onehot_val = obj.predict_onehot(np.array([[0,1]]))
predict_onehot_val = obj.predict_onehot(np.array([[1,1]]))'''

'''xor_onehot_data = make_onehot_t_data(xor_t_data, 2)
obj2 = LogisticRegression(x_data, xor_onehotencoding_data, 2, 6, 2, 1e-2, 120001)
obj2.train()

accuracy_rat, matched_list, notmatched_list = obj2.accuracy_onehot(x_data, xor_t_data)'''