import tensorflow as tf
import numpy as np

train_loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/diabetes_trainV2_normV2.csv', delimiter=',',
                         dtype=np.float32)

train_x_data = train_loaded_data[:, 0:-1]
train_t_data = np.round(train_loaded_data[:, [-1]])

test_loaded_data = np.loadtxt('C:/Users/user/PycharmProjects/DeepLearning/data/diabetes_testV2_normV2.csv', delimiter=',',
                         dtype=np.float32)

test_x_data = test_loaded_data[:, 0:-1]
test_t_data = np.round(test_loaded_data[:, [-1]])
delta = 1e-7

print("loaded_data", train_loaded_data.shape)
print("x_data", train_x_data.shape, "t_data", train_t_data.shape)

X = tf.placeholder(tf.float32, [None, train_x_data.shape[1]])
T = tf.placeholder(tf.float32, [None, train_t_data.shape[1]])

W2 = tf.Variable(tf.random_normal([8,1]))
b2 = tf.Variable(tf.random_normal([1]))

Z = tf.matmul(X, W2) + b2
y = tf.sigmoid(Z)

loss = -tf.reduce_mean(T*tf.log(y+delta) + (1-T)*tf.log(1-y+delta))

learning_rate = 1e-1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted = tf.cast(y>0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, T), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        y_val, loss_val, train_val = sess.run([y, loss, train], feed_dict={X:train_x_data, T:train_t_data})

        if step % 500 == 0:
            print("step", step, "loss_val", loss_val)

    y_val, predicted_val, accuarcy_val = sess.run([y, predicted, accuracy], feed_dict={X:test_x_data, T:test_t_data})

    print("y_val", y_val, "predicted_val", predicted_val)
    print("Accuracy_val", accuarcy_val)