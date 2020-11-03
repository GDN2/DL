import numpy as np
import tensorflow as tf
from datetime import datetime
import os
from tensorflow.examples.tutorials.mnist import input_data
import shutil
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True )

learning_rate = 1e-3
batch_size = 100
epochs = 10
input_nodes = 784
output_nodes = 10

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])
X_img = tf.reshape(X, [-1,28,28,1])

W2 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
b2 = tf.Variable(tf.random_normal([32]))

C2 = tf.nn.conv2d(X_img, W2, strides=[1,1,1,1], padding='SAME')
Z2 = tf.nn.relu(C2 + b2)
A2 = tf.nn.max_pool(Z2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
A2_flat = P2_flat = tf.reshape(A2, [-1,14*14*32])

W3 = tf.Variable(tf.random_normal([14*14*32, output_nodes]))
b3 = tf.Variable(tf.random_normal([output_nodes]))

Z3 = logits = tf.matmul(A2_flat, W3) + b3
y = A3 = tf.nn.softmax(Z3)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels = T))

optimizer= tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted = tf.equal(tf.argmax(A3,1), tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = datetime.now()

    saver = tf.train.Saver()
    mytrain = "./mytrain.ckpt"
    if os.path.exists(mytrain + ".meta"):
        saver.restore(sess, mytrain)
    else:
        for epoch in range(epochs):
            total_batchs = int(mnist.train.num_examples/batch_size)
            for step in range(total_batchs):
                batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
                loss_val, train_val = sess.run([loss, train], feed_dict={X:batch_x_data, T:batch_t_data})

                if step % 10 == 0:
                    print("epoch =", epoch, "step =", step, "loss_val =", loss_val)
        saver.save(sess, mytrain)

    end_time = datetime.now()
    print("Time spend", end_time - start_time)

    test_x_data = mnist.test.images
    test_t_data = mnist.test.labels

    A3_val, predicted_val, accuracy_val = sess.run([A3, predicted, accuracy], feed_dict = {X: test_x_data, T: test_t_data})
    print("\naccuracy_val", accuracy_val)
    print("\nAccuracy_val", accuracy_val)

    temp_list = []
    index_false_list = []
    np_false_count = np.zeros([10])

    for index in range(len(predicted_val)):
        if predicted_val[index] == False:
            temp_list.append(index)
            temp_list.append(np.argmax(test_t_data[index]))
            temp_list.append(np.argmax(A3_val[index]))
            index_false_list.append(temp_list)
            temp_list = []
            np_false_count[np.argmax(test_t_data[index])] += 1

    print(index_false_list)
    print(np_false_count)

    curr_dir = os.getcwd()
    now = datetime.now()
    dir_name = "MNIST"+str(now.year)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    os.chdir(dir_name)

    for index in range(len(index_false_list)):
        title = "lable = "+str(index_false_list[index][1])+" predicted = "+str(index_false_list[index][2])
        plt.title(title)
        img = test_x_data[index_false_list[index][0], :].reshape(28,28)
        plt.imshow(img, cmap='gray')
        fig = plt.gcf()
        fig.savefig(str(index_false_list[index][0])+'.png')

        if index%10 == 0:
            print(index, "image saved!")
    os.chdir(dir_name)


