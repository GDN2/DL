import numpy as np
from datetime import datetime

class data_preprocessing_csv:

    def __init__(self, data, dividing_percent = 0.3):
        tmp_index = 0
        for index in range(len(data)):
            if data[len(data) - 1 - index] == '/':
                tmp_index = len(data) - 1 - index
                break
        self.data_path = data[:tmp_index]
        self.data_name = data[tmp_index + 1:-4]
        self.path = self.data_path + '/' + self.data_name + '.csv'
        self.dividing_percent = dividing_percent # for training_data
        self.training_data_name = self.data_name + '_train.csv'
        self.test_data_name = self.data_name + '_test.csv'
        self.norm_data_name = self.data_name + '_norm.csv'
        self.onehot_data_name = self.data_name + '_t_onehot.csv'
        self.count = 0

        self.max_column = 0
        self.min_column = 0

    def distribution(self):
        training_path = self.data_path + '/' + self.training_data_name
        test_path = self.data_path + '/' + self.test_data_name
        training_list = []
        test_list = []

        loaded_data = np.loadtxt(self.path, delimiter=',', dtype=np.float32)
        number_test_data = int(len(loaded_data) * self.dividing_percent)

        random_list = np.arange(len(loaded_data))
        np.random.shuffle(random_list)

        for index in range(len(random_list)):
            if index < number_test_data:
                training_list.append(loaded_data[random_list[index]])
            else:
                test_list.append(loaded_data[random_list[index]])

        training_np = np.array(training_list).reshape(-1, loaded_data.shape[1])
        test_np = np.array(test_list).reshape(-1, loaded_data.shape[1])

        np.savetxt(training_path, training_np, delimiter=',')
        np.savetxt(test_path, test_np, delimiter=',')

        print("Data_preprocessing_distribution is done randomly")

    def normalization(self, t_column=-1, Logistic=False):
        norm_path = self.data_path + '/' + self.norm_data_name
        transposed_loaded_data = np.loadtxt(self.path, delimiter=',', dtype=np.float32, unpack=True)
        self.max_column = np.max(transposed_loaded_data, axis=1).reshape(transposed_loaded_data.shape[0], -1)
        self.min_column = np.min(transposed_loaded_data, axis=1).reshape(transposed_loaded_data.shape[0], -1)

        print(self.max_column.shape)

        transposed_norm_np = ((transposed_loaded_data - self.min_column) / (self.max_column - self.min_column + 1e-7) + 0.01) * 0.99
        if Logistic == True:
            transposed_norm_np[t_column] = transposed_loaded_data[t_column]

        print(transposed_norm_np.shape)
        print(transposed_norm_np[t_column])
        norm_np = transposed_norm_np.T
        print(norm_np.shape)
        np.savetxt(norm_path, norm_np, delimiter=',')

        print("Data_preprocessing_normalization is done")

    def denormalization(self, t_data, t_column=-1):
        t_data = np.array(t_data)
        t_data_denomalization = (t_data / 0.99 - 0.01) * (self.max_column[t_column] - self.min_column[t_column] + 1e-7) + \
                                self.min_column[t_column]
        return t_data_denomalization

    def make_onehot_t_data(self, t_data, type):
        size_matrix = len(t_data) * (type)  # type 개수 주의
        zero_matrix = np.zeros(size_matrix).reshape(len(t_data), (type)) + 0.01
        for index in range(len(t_data)):
            zero_matrix[index, int(t_data[index])] = 0.99
        return zero_matrix

    def save_onehot_t_data(self, t_index, type):
        onehot_path = self.data_path + '/' + self.onehot_data_name
        loaded_data = np.loadtxt(self.path, delimiter=',', dtype=np.float32)
        loaded_t_data = loaded_data[:, [t_index]]
        onehot_t_data = self.make_onehot_t_data(loaded_t_data, type)
        np.savetxt(onehot_path, onehot_t_data, delimiter=',')

        print("Data_preprocessing_make_and_save t_onehot data is done")


'''obj = data_preprocessing_csv('./data/mnist_train.csv')
obj.save_onehot_t_data(0, 10)'''

'''obj = data_preprocessing_csv('./data/mnist_train.csv')
obj.normalization(0, True)

obj = data_preprocessing_csv('./data/mnist_test.csv')
obj.normalization(0, True)'''

'''obj = data_preprocessing_csv('./data/ThoracicSurgery.csv', 0.4)
obj.normalization()
obj2= data_preprocessing_csv('./data/ThoracicSurgery_norm.csv')
obj2.distribution()'''