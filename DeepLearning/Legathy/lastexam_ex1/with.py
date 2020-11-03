import numpy as np

loaded_data = np.loadtxt('./regression_testdata_03.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:,:-1]
t_data = loaded_data[:,[-1]]

print(x_data)
print(t_data)
print(x_data.shape)
print(t_data.shape)