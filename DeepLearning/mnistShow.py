import numpy as np
import matplotlib.pyplot as plt
loadtxt_train = np.loadtxt('./data/mnist_train.csv', delimiter=',', dtype=np.float32)
print("done")
loadtxt_test = np.loadtxt('./data/mnist_test.csv', delimiter=',', dtype=np.float32)
# train 1901 test 9982 717
img = loadtxt_train[1901, 1:].reshape(28,28)
plt.imshow(img, cmap='gray')
plt.show()
img2 = loadtxt_test[9982, 1:].reshape(28,28)
plt.imshow(img2, cmap='gray')
plt.show()