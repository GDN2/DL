import tensorflow as tf
import numpy as np
import data_preprocessing as dp

obj = dp.data_preprocessing_csv('../data', 'diabetes')
obj.normalization(-1, True)
obj2 = dp.data_preprocessing_csv('../data', 'diabetes_norm')
obj2.distribution()
loaded_train_data = np.loadtxt('../data/diabetes_norm_test.csv', delimiter=',')
loaded_test_data = np.loadtxt('../data/diabetes_norm_train.csv', delimiter=',')
x_train_data = loaded_train_data[:,:-1]
t_train_data = loaded_train_data[:,[-1]]

x_test_data = loaded_test_data[:, :-1]
t_test_data = loaded_test_data[:, [-1]]

print("loaded_train_data.shape", loaded_train_data.shape)
print("x_train_data.shape", x_train_data.shape, "t_train_data.shape", t_train_data.shape)

print("loaded_test_data.shape", loaded_test_data.shape)
print("x_test_data.shape", x_test_data.shape, "t_test_data.shape", t_test_data.shape)

X = tf.placeholder(tf.float32, [None,8])
T = tf.placeholder(tf.float32, [None,1])

W = tf.Variable(tf.random_normal([8,1]))
b = tf.Variable(tf.random_normal([1]))

z = tf.matmul(X, W) + b
y = tf.sigmoid(z)
loss = -tf.reduce_mean(T*tf.log(y) + (1-T)*tf.log(1-y))

learning_rate = 1e-2
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted = tf.cast(y>0.5, dtype=tf.float32)
accuarcy = tf.reduce_mean(tf.cast(tf.equal(predicted, T), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        loss_val, _ = sess.run([loss, train], feed_dict={X: x_train_data, T: t_train_data})

        if step % 500 == 0:
            print("step =", step, ", loss_val =", loss_val)

    y_val, predicted_val, accuarcy_val = sess.run([y, predicted, accuarcy], feed_dict={X: x_test_data, T: t_test_data})

    print("\ny_val.shape =", y_val.shape, ", predicted_val =", predicted_val.shape)
    print("\nAccuarcy =", accuarcy_val)
