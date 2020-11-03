import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import shutil

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 1e-3
input_nodes = 784
output_nodes = 10
epochs = 30
batch_size = 100

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])

A1 = X_img = tf.reshape(X, [-1, 28, 28, 1])

W2 = tf.Variable(tf.random.normal([3,3,1,32], stddev=0.01))
b2 = tf.Variable(tf.random.normal([32]))

C2 = tf.nn.conv2d(A1, W2, strides=[1,1,1,1], padding='SAME')
Z2 = tf.nn.relu(C2 + b2)
A2 = P2 = tf.nn.max_pool(Z2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
A2_img = tf.reshape(A2, [-1, 14, 14, 32])

W3 = tf.Variable(tf.random.normal([3,3,32,32], stddev=0.01))
b3 = tf.Variable(tf.random.normal([32]))

C3 = tf.nn.conv2d(A2_img, W3, strides=[1,1,1,1], padding="SAME")
Z3 = tf.nn.relu(C3+b3)
A3 = P3 = tf.nn.max_pool(Z3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
A3_flat = P3_flat = tf.reshape(A3, [-1, 7,7,32])

W4  = tf.Variable(tf.random_normal([3,3,32,32], stddev=0.01))
b4 = tf.Variable(tf.random.normal([32]))

C4 = tf.nn.conv2d(A3_flat, W4, strides=[1,1,1,1], padding="SAME")
Z4 = tf.nn.relu(C4+b4)
A4 = tf.nn.max_pool(Z4, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
A4_flat = tf.reshape(A4, [-1,4*4*32])

W5 = tf.Variable(tf.random_normal([4*4*32, 10]))
b5 = tf.Variable(tf.random_normal([10]))

Z5 = logits = tf.matmul(A4_flat, W5) + b5
y = A5 = tf.nn.softmax(Z5)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=T))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted = tf.equal(tf.argmax(A5, 1), tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = datetime.now()

    for i in range(epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        for step in range(total_batch):
            batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
            loss_val, train_val = sess.run([loss, train], feed_dict={X: batch_x_data, T:batch_t_data})

            if step % 10 == 0:
                print("epochs =", i, ", step =", step, ", loss_val =", loss_val)
    end_time = datetime.now()
    print("\nelapsed time =", end_time - start_time)

    test_x_data = mnist.test.images
    test_t_data = mnist.test.labels

    A5_val, predicted_val, accuracy_val = sess.run([A5, predicted, accuracy], feed_dict = {X: test_x_data, T: test_t_data})
    print("\nAccuracy_val", accuracy_val)

    end_time = datetime.now()
    print("Time spen", end_time - start_time)

    temp_list = []
    index_label_false_list = []
    np_false_number = np.zeros([10])

    for index in range(len(predicted_val)):
        if predicted_val[index] == False:
            temp_list.append(index)
            temp_list.append(np.argmax(test_t_data[index]))
            temp_list.append(np.argmax(A5_val[index]))
            index_label_false_list.append(temp_list)
            temp_list = []
            np_false_number[np.argmax(test_t_data[index])] += 1
        else:
            pass
    print(index_label_false_list)
    print(np_false_number)
    #for index in range(len(index_label_false_list)):
    curr_dir = os.getcwd()
    print(os.getcwd())
    deeplearning_path = 'C:\\Users\\user\\PycharmProjects\\DeepLearning'

    first_file_name = 'mnist_false_figure'
    now = datetime.now()
    file_name = first_file_name + str(now.year)+'_' + str(now.month)+'_' + str(now.day)+'_' + str(now.hour)+'_' + str(now.minute)
    direct = deeplearning_path+'\\'+file_name
    if os.path.exists(direct):
        shutil.rmtree(direct)
    os.mkdir(direct)
    os.chdir(direct)

    for index in range(len(index_label_false_list)):
        plt.title("label = "+str(index_label_false_list[index][1])+" prediction = "+str(index_label_false_list[index][2]))
        img = test_x_data[index_label_false_list[index][0], :].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        fig = plt.gcf()
        fig.savefig(str(index_label_false_list[index][0])+'.png')
        if index % 10 == 0:
            print("index =", index, "images are saved!")
    os.chdir(curr_dir)