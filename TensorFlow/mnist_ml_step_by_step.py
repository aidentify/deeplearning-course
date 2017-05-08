__author__ = 'socurites'

import tensorflow as tf
import numpy as np

# Load MNIST datastes
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Explore MNIST datasets
mnist
mnist.train
mnist.train.images.shape
mnist.train.labels.shape
mnist.test
mnist.test.images.shape
mnist.test.labels.shape

# Show image of 3rd image
img_3 = mnist.train.images[2]
img_3.shape
img_3_reshaped = mnist.train.images[2].reshape((28, 28))
img_3_reshaped.shape
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.imshow(img_3_reshaped, cmap = cm.Greys)
plt.show()

# Show labe of 3rd image
mnist.train.labels[2]

# create variable for model: W * x + b
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define soft max funciton for output layer
y = tf.matmul(x, W) + b

# crate variable for real y
y_ = tf.placeholder(tf.float32, [None, 10])

# define loss: cross entropy
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# define optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# run training repeatedly within session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# evaluate training accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# cast boolean into float
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))