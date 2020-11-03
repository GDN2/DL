import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime
import shutil
import os
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.1
epochs = 1
batch_size = 100

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])

A1 = X_img = tf.reshape(X, [-1, 28, 28, 1]) #batch_size로 하면 test할 때 문제가 됨, test 시는 10000개를 보냄

W2 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
b2 = tf.Variable(tf.random_normal([32]))

C2 = tf.nn.conv2d(A1, W2, strides=[1,1,1,1], padding='SAME')
Z2 = tf.nn.relu(C2+b2)
A2 = P2 = tf.nn.max_pool(Z2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

A2_flat = P2_flat = tf.reshape(A2, [-1,14*14*32])

W3 = tf.Variable(tf.random_normal([14*14*32, 10], stddev=0.01))
b3 = tf.Variable(tf.random_normal([10]))

Z3 = logits = tf.matmul(A2_flat, W3) + b3
y = A3 = tf.nn.softmax(Z3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=T)) #logit은 y가 아니라 Z를 넣어야함 주의!

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted_val = tf.equal(tf.argmax(A3,1),tf.argmax(T,1)) # 그전에는 predicted였고 0.42114같은 값을 우선 0이나 1로 바꾸는 작업을 먼저하고 T랑 비교 여기서는 생략하고 바로 T랑 비교

accuracy = tf.reduce_mean(tf.cast(predicted_val, dtype=tf.float32))

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