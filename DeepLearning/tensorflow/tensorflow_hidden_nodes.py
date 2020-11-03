import tensorflow as tf
import numpy as np

loaded_train_data = np.loadtxt('../data/diabetes_norm_train.csv', delimiter=',')
loaded_test_data = np.loadtxt('../data/diabetes_norm_test.csv', delimiter=',')

train_x_data = loaded_train_data[:, :-1]
train_t_data = loaded_train_data[:, [-1]]

test_x_data = loaded_test_data[:, :-1]
test_t_data = loaded_test_data[:, [-1]]

X = tf.placeholder(tf.float32, [None, 8])
T = tf.placeholder(tf.float32, [None, 1])

W2 = tf.Variable(tf.random_normal([8, 6]))
b2 = tf.Variable(tf.random_normal([6]))

W3 = tf.Variable(tf.random_normal([6, 1]))
b3 = tf.Variable(tf.random_normal([1]))

Z2 = tf.matmul(X, W2) + b2
A2 = tf.sigmoid(Z2)

Z3 = tf.matmul(A2, W3) + b3
A3 = tf.sigmoid(Z3)
delta = 1e-7
loss = -tf.reduce_mean(T*tf.log(A3+delta)+(1-T)*tf.log(1-A3+delta))

learning_rate = 5e-1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)


predicted = tf.cast(A3 > 0.5 ,dtype = tf.float32)
accuarcy = tf.reduce_mean(tf.cast(tf.equal(predicted, T), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for index in range(10001):
        loss_val, train_val = sess.run([loss, train], feed_dict={X : train_x_data, T: train_t_data})

        if index % 500 == 0:
            print("index", index, "loss_val", loss_val)

    y_val, predicted_val, accuarcy_val = sess.run([A3, predicted, accuarcy], feed_dict={X: test_x_data, T: test_t_data})
    print("accuarcy ==", accuarcy_val)


