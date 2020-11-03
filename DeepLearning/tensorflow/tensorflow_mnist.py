import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("")
print("train.num =", mnist.train.num_examples, ", test.num =", mnist.test.num_examples, ", validation.num =", mnist.validation.num_examples)

print("type(mnist)", type(mnist))
print("type(mnist.train)", type(mnist.train))
print("type(mnist.train.images)", type(mnist.train.images))
print("type(mnist.train.labels)", type(mnist.train.labels))
print("mnist.train.images.shape", mnist.train.images.shape)
print("np.shape", np.shape(mnist.train.images))
print("mnist.train.images[0]", mnist.train.images[0])
print("mnist.train.labels[0]", mnist.train.labels[0])

learning_rate = 0.1
epochs = 100
batch_size = 100

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])

W2 = tf.Variable(tf.random_normal([input_nodes, hidden_nodes]))
b2 = tf.Variable(tf.random_normal([hidden_nodes]))
W3 = tf.Variable(tf.random_normal([hidden_nodes,output_nodes]))
b3 = tf.Variable(tf.random_normal([output_nodes]))

Z2 = tf.matmul(X, W2) + b2
A2 = tf.nn.relu(Z2)

Z3 = logits = tf.matmul(A2, W3) + b3
y = A3 = tf.nn.softmax(Z3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=T))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted = tf.equal(tf.argmax(A3, 1), tf.argmax(T, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = datetime.now()
    for epoch in range(epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        for step in range(total_batch):
            batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
            loss_val, train_val = sess.run([loss, train], feed_dict={X : batch_x_data, T : batch_t_data})
            if step % 100 == 0:
                print("epoch =", epoch, "step =", step, ", loss_val =", loss_val)

    test_x_data = mnist.test.images
    test_t_data = mnist.test.labels

    A3_val, predicted_val, accuracy_val = sess.run([A3, predicted, accuracy], feed_dict = {X: test_x_data, T: test_t_data})
    print("\nAccuracy_val", accuracy_val)

    end_time = datetime.now()
    print("Time spen", end_time - start_time)

    temp_list = []
    index_label_false_list = []

    for index in range(len(predicted_val)):
        if predicted_val[index] == False:
            temp_list.append(index)
            temp_list.append(np.argmax(test_t_data[index]))
            temp_list.append(np.argmax(A3_val[index]))
            index_label_false_list.append(temp_list)
            temp_list = []
        else:
            pass
    print(index_label_false_list)
    #for index in range(len(index_label_false_list)):
    curr_dir = os.getcwd()
    print(os.getcwd())
    deeplearning_path = 'C:\\Users\\user\\PycharmProjects\\DeepLearning'

    first_file_name = 'mnist_false_figure'
    now = datetime.now()
    file_name = first_file_name + str(now.year)+'_' + str(now.month)+'_' + str(now.day)+'_' + str(now.hour)+'_' + str(now.minute)
    os.mkdir(deeplearning_path+'\\'+file_name)
    os.chdir(deeplearning_path+'\\'+file_name)

    for index in range(len(index_label_false_list)):
        plt.title("label = "+str(index_label_false_list[index][1])+" prediction = "+str(index_label_false_list[index][2]))
        img = test_x_data[index_label_false_list[index][0], :].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        fig = plt.gcf()
        fig.savefig(str(index_label_false_list[index][0])+'.png')
        if index % 10 == 0:
            print("index =", index, "images are saved!")
    os.chdir(curr_dir)