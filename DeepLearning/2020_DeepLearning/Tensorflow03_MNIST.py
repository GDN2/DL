import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("train.num", mnist.train.num_examples)
print("test.num", mnist.test.num_examples)
print("validation.num", mnist.validation.num_examples)
print("type(mnist)", type(mnist))
print("type(mnist.train.images)", type(mnist.train.images))
print("mnist.train.images.shape", mnist.train.images.shape)
print("type(mnist.train.labels)", type(mnist.train.labels))
print("mnist.train.labels.shape", mnist.train.labels.shape)
for i in range(10): #one-hot incoding은 mnist.train.labels의 shape참고, 정규화 여부는 10번 쯤 찍어서 확인
    print("mnist.train.images["+str(i)+"]", mnist.train.images[i])
    print("mnist.train.labels["+str(i)+"]", mnist.train.labels[i])

learning_rate = 0.1
epochs = 5
batch_size = 100

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])

W2 = tf.Variable(tf.random_normal([input_nodes,hidden_nodes]))
b2 = tf.Variable(tf.random_normal([hidden_nodes]))

W3 = tf.Variable(tf.random_normal([hidden_nodes,output_nodes]))
b3 = tf.Variable(tf.random_normal([output_nodes]))

Z2 = tf.matmul(X, W2) + b2
A2 = tf.nn.relu(Z2)

Z3 = logits = tf.matmul(A2, W3) + b3
y = A3 = tf.nn.softmax(Z3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=T)) #logit은 y가 아니라 Z를 넣어야함 주의!

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

predicted_val = tf.equal(tf.argmax(A3,1),tf.argmax(T,1)) # 그전에는 predicted였고 0.42114같은 값을 우선 0이나 1로 바꾸는 작업을 먼저하고 T랑 비교 여기서는 생략하고 바로 T랑 비교

accuracy = tf.reduce_mean(tf.cast(predicted_val, dtype=tf.float32))

start = datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        for step in range(total_batch):
            batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
            loss_val, _ = sess.run([loss, train], feed_dict={X : batch_x_data, T : batch_t_data})
            if step % 100 == 0:
                print("step", step, "loss_val", loss_val)

    test_x_data = mnist.test.images
    test_t_data = mnist.test.labels

    accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, T:test_t_data})

    print(predicted_val)
    print("Accuracy", accuracy_val)

end = datetime.now()
print("Time", end-start)