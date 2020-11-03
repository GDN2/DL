import numpy as np
import matplotlib.pyplot as plt

loaded_train_data = np.loadtxt('./data/airplane_train_norm.csv', delimiter=',', dtype=np.float32, unpack=True)
loaded_test_data = np.loadtxt('./data/airplane_test_norm.csv', delimiter=',', dtype=np.float32, unpack=True)


plt.plot(loaded_train_data[0, :], loaded_train_data[-1,:], 'bo')
plt.show()
plt.plot(loaded_train_data[1, :], loaded_train_data[-1,:], 'go')
plt.show()

