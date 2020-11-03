import tensorflow as tf
import numpy as np
import data_preprocessing_v2 as dp_v2

obj = dp_v2.data_preprocessing_csv('../data/airplane_train.csv')
obj.normalization()

obj2 = dp_v2.data_preprocessing_csv('../data/airplane_test.csv')
obj2.normalization()

train_loaded_data = np.loadtxt('../data/airplane_train_norm.csv', delimiter=',')
train_x_data = train_loaded_data[:, :-1]
train_t_data = train_loaded_data[:, [-1]]

test_loaded_data = np.loadtxt('../data/airplane_test_norm.csv', delimiter=',')
test_x_data = test_loaded_data[:, :-1]
test_t_data = test_loaded_data[:, [-1]]

X = tf.placeholder(tf.float32, [None, 2])
T = tf.placeholder(tf.float32, [None, 1])

W2 = tf.Variable(tf.random_normal([2,6]))
b2 = tf.Variable(tf.random_normal([6]))

W3 = tf.Variable(tf.random_normal([6,1]))
b3 = tf.Variable(tf.random_normal([1]))

Z2 = tf.matmul(X, W2) + b2
A2 = tf.sigmoid(Z2) #Z2 정규화

Z3 = tf.matmul(A2, W3) + b3
y = Z3
loss = tf.reduce_mean(tf.square(T-y))

learning_rate = 1e-2
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1):
        for step in range(10001):
            loss_val, train_val = sess.run([loss, train], feed_dict = {X:train_x_data, T:train_t_data})
            if step % 500 == 0:
                print("step", step, "loss_val", loss_val)
    y_val = sess.run([y], feed_dict = {X:test_x_data, T:test_t_data})
    y_val = obj2.denormalization(y_val)
    test_t_data_rnomalization = obj2.denormalization(test_t_data)
    for index in range(y_val.shape[1]):
        print("y_val", y_val[0][index], "t_val", test_t_data_rnomalization[index],
              "error_rate", (test_t_data_rnomalization[index]-y_val[0][index])/test_t_data_rnomalization[index])