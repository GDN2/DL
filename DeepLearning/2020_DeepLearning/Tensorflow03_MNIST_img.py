
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime
import shutil
import os
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("train.num", mnist.train.num_examples)
print("test.num", mnist.test.num_examples)
print("validation.num", mnist.validation.num_examples)
print("type(mnist)", type(mnist))
print("type(mnist.train.images)", type(mnist.train.images))
print("mnist.train.images.shape", mnist.train.images.shape)
print("type(mnist.train.labels)", type(mnist.train.labels))
print("mnist.train.labels.shape", mnist.train.labels.shape)
#one-hot incoding은 mnist.train.labels의 shape참고, 정규화 여부는 10번 쯤 찍어서 확인
print("mnist.train.images["+str(1)+"]", mnist.train.images[1])
print("mnist.train.labels["+str(1)+"]", mnist.train.labels[1])

learning_rate = 0.1
epochs = 1
batch_size = 100

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])

W2 = tf.Variable(tf.random_normal([input_nodes,hidden_nodes]))
b2 = tf.Variable(tf.random_normal([hidden_nodes]))

W3 = tf.Variable(tf.random_normal([hidden_nodes,output_nodes]))
b3 = tf.Variable(tf.random_normal([output_nodes]))

Z2 = tf.matmul(X, W2) + b2
A2 = tf.nn.relu(Z2)

Z3 = logits = tf.matmul(A2, W3) + b3
y = A3 = tf.nn.softmax(Z3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=T)) #logit은 y가 아니라 Z를 넣어야함 주의!

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted_val = tf.equal(tf.argmax(A3,1),tf.argmax(T,1)) # 그전에는 predicted였고 0.42114같은 값을 우선 0이나 1로 바꾸는 작업을 먼저하고 T랑 비교 여기서는 생략하고 바로 T랑 비교

accuracy = tf.reduce_mean(tf.cast(predicted_val, dtype=tf.float32))

#predicted_val_list = tf.cast(predicted_val, dtype=tf.float32) #이렇게 두어서 predicted_val_list_val[index] == True 값을 넣어도 됨
#accuracy_val = np.argmax(A3, axis=1) # y를 np.argmax로 전처리
#T_val = np.argmax(T, axis=1) # T를 np.argmax로 전처리

start = datetime.now()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        for step in range(total_batch):
            batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
            loss_val, _ = sess.run([loss, train], feed_dict={X : batch_x_data, T : batch_t_data})
            if step % 100 == 0:
                print("step", step, "loss_val", loss_val)

    test_x_data = mnist.test.images
    test_t_data = mnist.test.labels

    y_val, predicted_val_list, accuracy_val = sess.run([y, predicted_val, accuracy], feed_dict={X: test_x_data, T:test_t_data})

    print(predicted_val_list)
    for i in range(10):
        print(np.argmax(y_val[i]))
    print("Accuracy", accuracy_val)

end = datetime.now()
print("Time", end-start)

def accuracy(x_data, t_data):
    matched_list = []
    temp_list = []
    unmatched_list = []

    for index in range(len(predicted_val_list)):
        y = y_val[index]

        if round(float(np.argmax(y))) == round(float(np.argmax(t_data[index]))):
                matched_list.append(index)
        else:
            temp_list = []
            temp_list.append(index)
            temp_list.append(int(round(float(np.argmax(t_data[index])))))
            temp_list.append(int(round(float(np.argmax(y)))))
            unmatched_list.append(temp_list)


    accuracy_rate = float(len(matched_list)/len(x_data))

    return accuracy_rate, matched_list, unmatched_list
accuracy_rate, matched_list, unmatched_list = accuracy(test_x_data,test_t_data)

cdir = os.getcwd()

now = datetime.now().strftime("%Y%m%d%H%M")
print(now)

path = cdir+'/'+'MNIST_'+str(now)
print(path)

if (os.path.exists(path) == True):
    print("Path is already exists")
    shutil.rmtree(path, True)
    print("Path is deleted")

os.mkdir(path)
print("Path is created", path)

os.chdir(path)
print(os.getcwd())

start = datetime.now()
for i in range(10): #10개만
    img = test_x_data[unmatched_list[i][0], :].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title("index = " + str(unmatched_list[i][0]) +", label = " + str(unmatched_list[i][1]) + ", predict = " + str(unmatched_list[i][2]))
    
    plt.savefig(str(i+1)+'.png')
    plt.show()
    
    if (i % 10 == 0):
        print(i, "png images", "are saved!")

end = datetime.now()
print("Time", end - start)
print("All images are saved by png files")
os.chdir(cdir)
print(os.getcwd())






