import numpy as np

class Normalizer:

    def __init__(self, fullpath):

        self.fullpath = fullpath
        self.directory = ''
        self.filename = ''
        self.extension = ''
        self.loaded_data = np.zeros(1, dtype=np.float32)
        self.x_data = np.zeros(1, dtype=np.float32)
        self.t_data = np.zeros(1, dtype=np.float32)

        extension = self.fullpath.split('.')[-1]
        self.extension = ''.join(extension)

        filename = self.fullpath.split('/')[-1]
        filename = ''.join(filename).split('.')[0]
        self.filename = ''.join(filename) # 파일이름 '/' 이후 마지막 부분 확장자 제외

        directory = self.fullpath.split('/')[:-1]
        self.directory = '/'.join(directory) # 마지막 / 이전까지

        print("path : ", fullpath)
        print("directory : ", self.directory)
        print("filename : ", self.filename)
        print("extension : ", self.extension)

    def maxMinNormaizer(self, regression = 'logistic'):

        path = self.directory+'/'+self.filename+'_normV2'+'.'+self.extension
        print(path)

        if (self.extension == 'csv'):
            self.loaded_data = np.loadtxt(self.fullpath, delimiter=',', dtype=np.float32)

        delta_x = 1e-4

        maxXT = np.max(self.loaded_data, axis=0).reshape(1,-1)
        minXT = np.min(self.loaded_data, axis=0).reshape((1,-1))

        loaded_data_norm = ((self.loaded_data - minXT)/(maxXT - minXT + delta_x) + delta_x) * (1 - delta_x)

        np.savetxt(path, loaded_data_norm , fmt='%10.7f', delimiter=',')
        loaded_data = np.loadtxt(path, delimiter=',', dtype=np.float32)

        print("maxMinNormalizing is finished")

    def shuffler(self, ratio):
        train_path = self.directory+'/'+self.filename+'_trainV2'+'.'+self.extension
        test_path = self.directory+'/'+self.filename+'_testV2'+'.'+self.extension

        if (self.extension == 'csv'):
            self.loaded_data = np.loadtxt(self.fullpath, delimiter=',', dtype=np.float32)

        train_data_size = int(len(self.loaded_data) * ratio)
        test_data_size = len(self.loaded_data) - train_data_size

        shuffleSeed = np.arange(len(self.loaded_data))
        np.random.shuffle(shuffleSeed)
        loaded_data = self.loaded_data[shuffleSeed]

        train_data = loaded_data[:train_data_size]
        test_data = loaded_data[train_data_size:]

        np.savetxt(train_path, train_data , fmt='%10.7f', delimiter=',')
        np.savetxt(test_path, test_data , fmt='%10.7f', delimiter=',')

        print("Data Size", len(loaded_data))
        print("Train Data Size", train_data_size)
        print("test Data Size", test_data_size)
        print("Shuffling have been finished")

    def shuffleMaxMinNormalizer(self, ratio):
        self.shuffler(ratio) #shuffle 사용하고 저장하기위해 V2이름 필요함
        train_path = self.directory+'/'+self.filename+'_trainV2'+'.'+self.extension
        test_path = self.directory+'/'+self.filename+'_testV2'+'.'+self.extension
        trainObj = Normalizer(train_path)
        testObj = Normalizer(test_path)

        trainObj.maxMinNormaizer()
        testObj.maxMinNormaizer()

    def fairDivider(self, answer_index):
        divide_path = self.directory+'/'+self.filename+'_divide'+'.'+self.extension

        if (self.extension == 'csv'):
            self.loaded_data = np.loadtxt(self.fullpath, delimiter=',', dtype=np.float32)

        False_index_list = np.array([i for i in range(len(self.loaded_data)) if self.loaded_data[i][answer_index] == 0])
        True_index_list = np.array([i for i in range(len(self.loaded_data)) if self.loaded_data[i][answer_index] == 1])

        print("False Index Size", len(False_index_list))
        print("True Index Size", len(True_index_list))

        listSeed = np.arange(len(False_index_list)) if len(False_index_list) <= len(True_index_list) else np.arange(len(True_index_list))
        print("Using list Size", len(listSeed))

        print("False_index_list", False_index_list)
        print("True_index_list", True_index_list)

        np.random.shuffle(False_index_list)
        np.random.shuffle(True_index_list)

        False_index_list = list(False_index_list[listSeed])
        True_index_list = list(True_index_list[listSeed])

        print("False_index_list", False_index_list)
        print("True_index_list", True_index_list)

        divideSeed = np.array(False_index_list + True_index_list)
        np.random.shuffle(divideSeed)
        divided_data = self.loaded_data[divideSeed]

        print("divideSeed", divideSeed)

        np.savetxt(divide_path, divided_data, fmt='%10.7f', delimiter=',')

'''
try:
    shuffler = Normalizer('C:/Users/user/PycharmProjects/DeepLearning/data/wine.csv')
    shuffler.shuffleMaxMinNormalizer(0.7)

except Exception as err:
    print(str(err))
    print("Error occur")
'''

try:
    shuffler = Normalizer('C:/Users/user/PycharmProjects/DeepLearning/data/airplane_train.csv')
    shuffler.shuffler(0.5)
except Exception as err:
    print(str(err))
    print("Error occur")











