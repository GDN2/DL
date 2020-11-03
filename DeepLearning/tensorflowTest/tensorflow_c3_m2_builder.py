import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import shutil
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

learning_rate = 1e-3
batch_size = 100
epochs = 30

X = tf.placeholder(tf.float32, [None, 784], name='x')
T = tf.placeholder(tf.float32, [None, 10], name='y')
A1 = X_img = tf.reshape(X, [-1,28,28,1])

W2 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
b2 = tf.Variable(tf.random_normal([32]))

C2 = tf.nn.conv2d(A1, W2, strides=[1,1,1,1], padding='SAME')
Z2 = tf.nn.relu(C2+b2)
A2 = tf.nn.max_pool(Z2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
A2_flat = tf.reshape(A2, [-1,14,14,32])

W3 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
b3 = tf.Variable(tf.random_normal([64]))

C3 = tf.nn.conv2d(A2_flat, W3, strides=[1,1,1,1], padding='SAME')
Z3 = tf.nn.relu(C3+b3)
A3 = tf.nn.max_pool(Z3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
A3_flat = tf.reshape(A3, [-1,7,7,64])

W4 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
b4 = tf.Variable(tf.random_normal([128]))

C4 = tf.nn.conv2d(A3_flat, W4, strides=[1,1,1,1], padding='SAME')
Z4 = tf.nn.relu(C4+b4)
A4 = Z4
A4_flat = P4_flat = tf.reshape(A4, [-1,7*7*128])

W5 = tf.Variable(tf.random_normal([7*7*128,10]))
b5 = tf.Variable(tf.random_normal([10]))

Z5 = logits = tf.matmul(A4_flat, W5, name='h') + b5
h = tf.identity(Z5, name='h')

y = A5 = tf.nn.softmax(Z5)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels = T))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted = tf.equal(tf.argmax(A5,1), tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = datetime.now()

    builder_folder_name = '../mnist_builder'
    if not os.path.exists(builder_folder_name):
        os.mkdir(builder_folder_name)
    path_name = builder_folder_name+'/mytrain.ckpt'
    if os.path.exists(path_name+'.meta'):
        pass
    else:
        for epoch in range(epochs):
            total_batchs = int(mnist.train.num_examples/batch_size)
            for step in range(total_batchs):
                batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
                loss_val, train_val = sess.run([loss, train], feed_dict={X:batch_x_data, T:batch_t_data})

                if step % 10 == 0:
                    print("epoch =", epoch, "step =", step, "loss_val =", loss_val)

        builder = tf.saved_model.builder.SavedModelBuilder("/tmp/fromPython")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        builder.save()
    end_time = datetime.now()
    print("Time spend =", end_time-start_time)

    start_time = datetime.now()

    test_x_data = mnist.test.images
    test_t_data = mnist.test.labels
    start_time = datetime.now()
    A5_val, predicted_val, accuracy_val = sess.run([A5, predicted, accuracy], feed_dict={X:test_x_data, T:test_t_data})
    print("Accuracy =", accuracy_val)

    end_time = datetime.now()
    print("Time spend =", end_time-start_time)

    start_time = datetime.now()

    temp_list = []
    false_index_list = []
    np_false_count = np.zeros([10])

    for index in range(len(predicted_val)):
        if predicted_val[index] == False:
            temp_list.append(index)
            temp_list.append(np.argmax(test_t_data[index]))
            temp_list.append(np.argmax(A5_val[index]))
            false_index_list.append(temp_list)
            temp_list = []
            np_false_count[np.argmax(test_t_data[index])] += 1

    curr_dir = os.getcwd()
    now = datetime.now()
    folder_first_name = "../MNIST_FALSE_LIST"
    folder_name = folder_first_name + str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)
    direct = folder_name
    if os.path.exists(direct):
        shutil.rmtree(direct)
    os.mkdir(direct)
    os.chdir(direct)

    for index in range(len(false_index_list)):
        title = "label = "+str(false_index_list[index][1])+"predicted = "+str(false_index_list[index][2])
        plt.title(title)
        img = test_x_data[false_index_list[index][0],:].reshape(28,28)
        plt.imshow(img, cmap='gray')
        fig = plt.gcf()
        fig.savefig(str(false_index_list[index][0])+'.png')
        if index % 10 == 0:
            print(index, "are saved!")
    os.chdir(curr_dir)

    end_time = datetime.now()
    print("Time spend =", end_time-start_time)
