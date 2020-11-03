import tensorflow as tf
import numpy as np
from datetime import datetime

train_loaded_data = np.loadtxt('../data/ThoracicSurgery_norm_train.csv', delimiter=',')
test_loaded_data = np.loadtxt('../data/ThoracicSurgery_norm_test.csv', delimiter=',')

train_x_data = train_loaded_data[:, :-1]
train_t_data = train_loaded_data[:, [-1]]

test_x_data = test_loaded_data[:, :-1]
test_t_data = test_loaded_data[:, [-1]]

input_nodes = train_x_data.shape[1]
hidden_nodes = 6
output_nodes = 1

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])

W2 = tf.Variable(tf.random_normal([input_nodes, hidden_nodes]))
b2 = tf.Variable(tf.random_normal([hidden_nodes]))

W3 = tf.Variable(tf.random_normal([hidden_nodes, output_nodes]))
b3 = tf.Variable(tf.random_normal([output_nodes]))

Z2 = tf.matmul(X, W2) + b2
A2 = tf.sigmoid(Z2)

Z3 = tf.matmul(A2, W3) + b3
A3 = tf.sigmoid(Z3)
delta = 1e-7
loss = -tf.reduce_mean(T*tf.log(A3+delta)+ (1-T)*tf.log(1-A3+delta))

learning_rate = 1e-2
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted = tf.cast(A3>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, T), dtype=tf.float32))

epochs = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = datetime.now()
    for epoch in range(epochs):
        for step in range(10001):
            loss_val, _ = sess.run([loss, train], feed_dict={X : train_x_data, T : train_t_data})

            if step % 500 == 0:
                print("step =", step, "loss_val", loss_val )
    y_val, predicted_val, accuracy_val = sess.run([A3, predicted, accuracy], feed_dict={X: test_x_data, T : test_t_data})
    end_time = datetime.now()
    print("accuracy =", accuracy_val, "Time spend", end_time - start_time)
