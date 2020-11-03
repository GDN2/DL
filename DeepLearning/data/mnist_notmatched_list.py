import numpy as np
import matplotlib.pyplot as plt

loaded_txt = np.loadtxt('./mnist_test.csv', delimiter=',', dtype=int)
img = loaded_txt[33,1:].reshape(28,28)

x = np.arange(0, 28, 1)

plt.imshow(img, cmap='gray')
plt.show()