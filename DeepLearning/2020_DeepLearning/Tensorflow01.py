import tensorflow as tf
import numpy as np

loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/data-01.csv', delimiter=',',
                         dtype=np.float32)

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

print("x_data.shape", x_data.shape)
print("t_data.shape", t_data.shape)

W = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

X = tf.placeholder(tf.float32, [None, 3])
T = tf.placeholder(tf.float32, [None, 1])

y = tf.matmul(X, W) + b
loss = tf.reduce_mean(tf.square(y-T))

learning_rate = 1e-5

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(8001):
        y_val, loss_val, train_val = sess.run([y, loss, train], feed_dict={X: x_data, T: t_data})

        if step % 400 == 0:
            print("step", step, ",loss_val", loss_val)

    print("prediction is", sess.run(y, feed_dict={X:[[100,98,81]]}))

