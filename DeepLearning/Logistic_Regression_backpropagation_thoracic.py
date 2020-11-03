import numpy as np
from datetime import datetime

class LogisticRegression:
    def __init__(self, input_nodes, hidden_nodes, hidden_nodes2, output_nodes, learning_rate = 1e-2):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.hidden_nodes2 = hidden_nodes2
        self.output_nodes = output_nodes

        self.b4 = np.random.rand(output_nodes)
        self.W4 = np.random.randn(hidden_nodes2, output_nodes) / np.sqrt(self.hidden_nodes2/2)
        self.b3 = np.random.rand(hidden_nodes2)
        self.W3 = np.random.randn(hidden_nodes, hidden_nodes2) / np.sqrt(self.hidden_nodes/2)
        self.b2 = np.random.rand(hidden_nodes)
        self.W2 = np.random.randn(input_nodes,hidden_nodes) / np.sqrt(self.input_nodes/2)

        self.learning_rate = learning_rate

        self.A4 = np.zeros([1, output_nodes])
        self.Z4 = np.zeros([1, output_nodes])
        self.A3 = np.zeros([1, hidden_nodes2])
        self.Z3 = np.zeros([1, hidden_nodes2])  # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.A2 = np.zeros([1, hidden_nodes])
        self.Z2 = np.zeros([1, hidden_nodes]) # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.A1 = np.zeros([1, input_nodes])
        self.Z1 = np.zeros([1, input_nodes])

        self.target_data = np.zeros([1,output_nodes])
        self.input_data = np.zeros([1,input_nodes])

    def sigmoid(self, z):
        return (1/(1+np.exp(-z))) # exp에는 delta_x필요 없음

    def loss_func(self):
        delta = 1e-7

        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2 # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3 # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.A3 = self.sigmoid(self.Z3)

        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.sigmoid(self.Z4)

        E = -np.sum(self.target_data * np.log(self.A4 + delta) + (1 - self.target_data) * np.log(1 - self.A4 + delta))

        return E

    def error_val(self):
        delta = 1e-7

        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2 # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3 # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
        self.A3 = self.sigmoid(self.Z3)

        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.sigmoid(self.Z4)

        E = -np.sum(self.target_data * np.log(self.A4 + delta) + (1 - self.target_data) * np.log(1 - self.A4 + delta))

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
           
            grad[idx] = (f1-f2)/(2*delta_x)# 분자에 괄호
            
            x[idx] = tmp_val
            it.iternext()
        
        return grad #return indent while에 맞추기

    def train(self, input_data, target_data):

        self.input_data = input_data
        self.target_data = target_data
        self.loss_func() # 이것을 안 해주면 W랑 b는 변하는데 error_val이 안 변하므로 모든 의미가 사라짐 , A랑 Z가 안 변함

        loss4 = (self.A4-self.target_data)*self.A4*(1-self.A4)
        loss3 = np.dot(loss4, self.W4.T)*self.A3*(1-self.A3)
        loss2 = np.dot(loss3, self.W3.T)*self.A2*(1-self.A2)

        self.b4 = self.b4 - self.learning_rate*loss4
        self.W4 = self.W4 - self.learning_rate*np.dot(self.A3.T, loss4)


        self.b3 = self.b3 - self.learning_rate*loss3
        self.W3 = self.W3 - self.learning_rate*np.dot(self.A2.T,loss3)


        self.b2 = self.b2 - self.learning_rate*loss2
        self.W2 = self.W2 - self.learning_rate*np.dot(self.A1.T, loss2)

    def predict(self, input_data):
        self.Z2 = np.dot(input_data, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.sigmoid(self.Z4)

        result = 0

        if self.A4 > 0.5:
            result = 1
            return self.A4, result # return indent위치 주의하고 마지막에 한 번 더 return 하지 말기
        elif self.A4 < 0.5:
            result = 0
            return self.A4, result

    def predict_onehot(self,input_data):
        self.Z2 = np.dot(input_data, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.sigmoid(self.Z4)

        return np.argmax(self.A4, axis=1)

    def accuracy(self,x_data,t_data):
        matched_list = []
        notmatched_list = []
        index_label_list = []
        temp_list = []
        accuracy_rate = 0
        roc_rate = 0
        roc_t_t = 0
        roc_f_f = 0
        roc_f_t = 0
        roc_t_f = 0

        for index in range(len(x_data)):
            (real_val, logical_val) = self.predict(np.array(x_data[index], ndmin = 2))
            if logical_val == int(np.rint(t_data[index])):
                accuracy_rate += 1
                matched_list.append(index)

                if logical_val == 1:
                    roc_t_t += 1
                elif logical_val == 0:
                    roc_f_f += 1
            else:
                notmatched_list.append(index)
                temp_list.append(index)
                temp_list.append(int(np.rint(t_data[index])))
                temp_list.append(logical_val)
                index_label_list.append(temp_list)
                temp_list = []

                if logical_val == 1:
                    roc_f_t += 1
                elif logical_val == 0:
                    roc_t_f += 1

        accuracy_rate = accuracy_rate/len(x_data)
        roc_t = roc_t_t / (roc_t_t + roc_t_f)
        roc_f = roc_f_f / (roc_f_f + roc_f_t)
        roc_rate = roc_t * roc_f

        print("matched_list ==>", matched_list)
        print("notmatched_list ==>", notmatched_list)
        print("index_lable_list ==>", index_label_list)
        print("accuracy_rate ==>", accuracy_rate)
        print("ROC rate==>", roc_rate)

        return accuracy_rate, matched_list, notmatched_list, index_label_list, roc_rate

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

    def accuracy_onehot(self,x_data, t_data, max=1):
        matched_list = []
        notmatched_list = []
        temp_list = []
        index_label_list = []
        accuracy_rate = 0
        count = 0

        for index in range(len(t_data)):
            argmax = int(self.predict_onehot(np.array(x_data[index], ndmin=2)))
            if count < 30:
                print(argmax, np.rint(t_data[index]*max))
                count += 1

            if argmax == np.rint(t_data[index]*max):
                accuracy_rate += 1
                matched_list.append(index)
            else:
                temp_list.append(index)
                temp_list.append(int(np.rint(t_data[index]*max)))
                temp_list.append(argmax)
                index_label_list.append(temp_list)
                temp_list = []
                notmatched_list.append(index)

        accuracy_rate = accuracy_rate/len(x_data)

        print("matched_list =>", matched_list)
        print("notmatched_list =>", notmatched_list)
        print("accuracy_rate =>", accuracy_rate)

        return accuracy_rate, matched_list, notmatched_list, index_label_list

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


processing_start_time = datetime.now()

start_time = datetime.now()
train_loaded_data = np.loadtxt('./data/mnist_train_norm.csv' , delimiter = ',', dtype = np.float32)
train_data = train_loaded_data[:, 1:]
end_time = datetime.now()
print("Processing time ==>", end_time-start_time)
print("process 1 is done")

start_time = datetime.now()
train_onehot = np.loadtxt('./data/mnist_train_t_onehot.csv' , delimiter = ',', dtype = np.float32)
end_time = datetime.now()
print("Processing time ==>", end_time-start_time)
print("process 2 is done")

start_time = datetime.now()
test_loaded_data = np.loadtxt('./data/mnist_test_norm.csv' , delimiter = ',', dtype = np.float32)
test_data = test_loaded_data[:, 1:]
test_t_data = test_loaded_data[:, [0]]
end_time = datetime.now()
print("Processing time ==>", end_time-start_time)
print("process 3 is done. Data preprocessing is done.")
processing_end_time = datetime.now()
print("Processing Total time ==>", processing_end_time-processing_start_time)

input_nodes = train_data.shape[1]
hidden_nodes = 100
hidden_nodes2 = 10
output_nodes = 10
epochs = 1

obj = LogisticRegression(input_nodes, hidden_nodes, hidden_nodes2, output_nodes, 0.5)
start_time = datetime.now()
for epoch in range(epochs):
    for index in range(len(train_data)):
        #input_data = (train_loaded_data[index, 1:]/255 * 0.99) + 0.01
        #target_data = np.zeros(output_nodes) + 0.01
        #target_data[int(train_loaded_data[index,0])] = 0.99
        input_data = train_data[index]
        target_data = train_onehot[index]
        obj.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2))
        if index % 400 == 0:
            print("index =", index, "error_val =", obj.loss_func())
    if epoch % 5 == 0:
        print("epoch =", epoch, "error_val =", obj.loss_func())
end_time = datetime.now()
print("Time spend ==>", end_time - start_time)

accuracy_rat, matched_list, notmatched_list, index_label_list = obj.accuracy_onehot(test_data, test_t_data)


'''
loaded_data = np.loadtxt('./data/ThoracicSurgery_norm.csv' , delimiter = ',', dtype = np.float32)

train_loaded_data = np.loadtxt('./data/ThoracicSurgery_norm_train.csv' , delimiter = ',', dtype = np.float32)
train_data = train_loaded_data[:, :-1]
train_t_data = train_loaded_data[:, [-1]]

test_loaded_data = np.loadtxt('./data/ThoracicSurgery_norm_test.csv' , delimiter = ',', dtype = np.float32)
test_data = test_loaded_data[:, :-1]
test_t_data = test_loaded_data[:, [-1]]

input_nodes = train_data.shape[1]
hidden_nodes = 100
hidden_nodes2 = 100
output_nodes = 1
epochs = 100

obj = LogisticRegression(input_nodes, hidden_nodes, hidden_nodes2, output_nodes, 2e-2)
start_time = datetime.now()
for epoch in range(epochs):
    for index in range(len(train_data)):
        #input_data = (train_loaded_data[index, 1:]/255 * 0.99) + 0.01
        #target_data = np.zeros(output_nodes) + 0.01
        #target_data[int(train_loaded_data[index,0])] = 0.99
        input_data = train_data[index]
        target_data = train_t_data[index]
        obj.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2))
    if epoch % 5 == 0:
        print("epoch =", epoch, "error_val =", obj.loss_func())
end_time = datetime.now()
print("Time spend ==>", end_time - start_time)


accuracy_rate, matched_list, notmatched_list, index_label_list, roc_rate = obj.accuracy(test_data, test_t_data)
'''
'''obj = LogisticRegression(x_data, xor_t_data, 2, 8, 1, 1e-2, 100001)
obj.train()
accuracy_rat, matched_list, notmatched_list, index_label_list, roc_rate = obj.accuracy(x_data, xor_t_data)'''

'''loaded_data = np.loadtxt('./data/mnist_train.csv', delimiter = ',', dtype = np.float32)
training_data_mnist = loaded_data[:, 1:]
training_t_data


predict_onehot_val = obj.predict_onehot(np.array([[0,0]]))
predict_onehot_val = obj.predict_onehot(np.array([[1,0]]))
predict_onehot_val = obj.predict_onehot(np.array([[0,1]]))
predict_onehot_val = obj.predict_onehot(np.array([[1,1]]))

xor_onehot_data = make_onehot_t_data(xor_t_data, 2)
obj2 = LogisticRegression(x_data, xor_onehotencoding_data, 2, 6, 2, 1e-2, 120001)
obj2.train()

accuracy_rat, matched_list, notmatched_list = obj2.accuracy_onehot(x_data, xor_t_data)'''