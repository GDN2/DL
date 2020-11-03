import numpy as np

try:
    train_loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/diabetes_norm3.csv',
                                   delimiter=',', dtype=np.float32)

except Exception as err:
    print("File을 찾을 수 없습니다.")
    print(str(err))


