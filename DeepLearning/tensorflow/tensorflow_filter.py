import numpy as np

three_img = np.array([[0,0,1,1,0,0],[0,1,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,1,1,0]])

one_img = np.array([[0,0,0,1,0,0],[0,0,1,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0]])

test_img = np.array([[0,0,0,0,0,0],[0,1,2,3,0,0],[0,0,1,2,3,0],[0,3,0,1,2,0],[0,2,3,0,1,0],[0,0,0,0,0,0]])

filter6 = np.array([[2,0,1],[0,1,2],[1,0,2]])

print(three_img)
print(one_img)

print(test_img)
print(filter6)

filter1 = np.array([[0,0,0],[1,1,1],[0,0,0]]) # horizon
filter2 = np.array([[0,1,0],[0,1,0],[0,1,0]]) # vertical
filter3 = np.array([[1,0,0],[0,1,0],[0,0,1]]) # reverse slash
filter4 = np.array([[0,0,1],[0,1,0],[1,0,0]]) # slash
filter5 = np.array([[0,1,0],[1,0,1],[0,1,0]]) # crystal

print(filter1)
print(filter2)
print(filter3)
print(filter4)
print(filter5)

total_img = test_img
filter_img = filter6 # crystal
padding = 0
stride = 1
b = 1
index_range = int((total_img.shape[0] + 2*padding - filter_img.shape[0])/stride + 1)
filtered_list = []
temp_list = []
pooling_filtered_list = []

for i in range(index_range):
    for j in range(index_range):
        np_sum = np.sum(total_img[0+i*stride : filter_img.shape[0]+i*stride, 0+j*stride : filter_img.shape[1]+j*stride]*filter_img) + b
        filtered_list.append(np_sum)
filtered_img = np.array(filtered_list).reshape(index_range,-1)
print("filtered_img\n", filtered_img)

pooling_size = 2
index_range2 = int(filtered_img.shape[0] / pooling_size)

for i in range(index_range2):
    for j in range(index_range2):
        print("patial_filtered_img\n", filtered_img[0+i*pooling_size : 2+i*pooling_size, 0+j*pooling_size :2+j*pooling_size])
        np_max = np.max(filtered_img[0+i*pooling_size : 2+i*pooling_size, 0+j*pooling_size :2+j*pooling_size].reshape(1,-1))
        print("np_max =",np_max)
        pooling_filtered_list.append(np_max)
print(pooling_filtered_list)
pooling_filtered_img = np.array(pooling_filtered_list).reshape(index_range2,-1)
print("pooling_flitered_img\n", pooling_filtered_img)
