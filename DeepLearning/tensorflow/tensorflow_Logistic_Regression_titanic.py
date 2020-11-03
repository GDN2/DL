import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

loaded_train_data = pd.read_csv('../data/train.csv').values
loaded_test_data = pd.read_csv('../data/test.csv').values
loaded_test_t_data = pd.read_csv('../data/gender_submission.csv').values


x_train_data = loaded_train_data[:, [2, 4, 5, 6, 7, 11]]
t_train_data = loaded_train_data[:, [1]]
for index in range(len(x_train_data)):
    if str(x_train_data[index][1]) == 'male':
        x_train_data[index][1] = 1
    else:
        x_train_data[index][1] = 0

for index in range(len(x_train_data)):
    if np.isnan(x_train_data[index][2]):
        x_train_data[index][2] = 0
    else:
        pass

for index in range(len(x_train_data)):
    if str(x_train_data[index][5]) == 'C':
        x_train_data[index][5] = 1
    elif str(x_train_data[index][5]) == 'Q':
        x_train_data[index][5] = 2
    elif str(x_train_data[index][5]) == 'S':
        x_train_data[index][5] = 3
    else:
        x_train_data[index][5] = 0

x_test_data = loaded_test_data[:, [1, 3, 4, 5, 6, 10]]
t_test_data = loaded_test_t_data[:, [1]]

for index in range(len(x_test_data)):
    if str(x_test_data[index][1]) == 'male':
        x_test_data[index][1] = 1
    else:
        x_test_data[index][1] = 0

for index in range(len(x_test_data)):
    if np.isnan(x_test_data[index][2]):
        x_test_data[index][2] = 0
    else:
        pass

for index in range(len(x_test_data)):
    if str(x_test_data[index][5]) == 'C':
        x_test_data[index][5] = 1
    elif str(x_test_data[index][5]) == 'Q':
        x_test_data[index][5] = 2
    elif str(x_test_data[index][5]) == 'S':
        x_test_data[index][5] = 3
    else:
        x_test_data[index][5] = 0


print("loaded_train_data.shape", loaded_train_data.shape)
print("x_train_data.shape", x_train_data.shape, "t_train_data.shape", t_train_data.shape)

print("loaded_test_data.shape", loaded_test_data.shape)
print("x_test_data.shape", x_test_data.shape, "t_test_data.shape", t_test_data.shape)

X = tf.placeholder(tf.float32, [None,6])
T = tf.placeholder(tf.float32, [None,1])

W = tf.Variable(tf.random_normal([6,1]))
b = tf.Variable(tf.random_normal([1]))

delta = 1e-7
z = tf.matmul(X, W) + b
y = tf.sigmoid(z)
loss = -tf.reduce_mean(T*tf.log(y+delta) + (1-T)*tf.log(1-y+delta))

learning_rate = 1e-2
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted = tf.cast(y>0.5, dtype=tf.float32)
accuarcy = tf.reduce_mean(tf.cast(tf.equal(predicted, T), dtype=tf.float32))

start_time = datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5500):
        loss_val, _ = sess.run([loss, train], feed_dict={X: x_train_data, T: t_train_data})

        if step % 500 == 0:
            print("step =", step, ", loss_val =", loss_val)

    y_val, predicted_val, accuarcy_val = sess.run([y, predicted, accuarcy], feed_dict={X: x_test_data, T: t_test_data})

    print("\ny_val.shape =", y_val.shape, ", predicted_val =", predicted_val.shape)
    print("\nAccuarcy =", accuarcy_val)
end_time = datetime.now()
print("Time spend ==>", end_time - start_time)
