import matplotlib.pyplot as plt
import numpy as np


x_data = [-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
y_data = [-8,-13,0,3,6,-1,-5,-7,1,8,7,12,13]

plt.title('line plot')
plt.grid()
plt.scatter(x_data, y_data, color='g')
plt.show()