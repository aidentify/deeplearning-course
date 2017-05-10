#coding:utf-8
"""
MNIST classification in low level API

@author socurites@gmail.com
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import mnist_data as mnist_data
import math
from random import shuffle

'''
훈련 옵션
'''
# 미니배치 사이즈
batch_size = 100
# 학습률
learning_rate = 0.1
# 훈련 스텝
step_num = 2000

'''
네트워크 옵션
'''
image_size = 28
class_num = 10


'''
데이터 로드 (훈련용 / 평가용)
'''

# 데이터 로드
path = '../data/MNIST/'
(train_label, train_img) = mnist_data.read_data(
    path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_label, val_img) = mnist_data.read_data(
    path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

# 10개의 이미지와 레이블 plotting
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(train_img[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_label[0:10]))

# 타입 변환
train_label = train_label.astype(np.int64)
train_img = train_img.astype(np.float32) / 255
val_label = val_label.astype(np.int64)
val_img = val_img.astype(np.float32) / 255


'''
네트워크 정의
CNN(Convolution Neural Network)
'''
# 입력 레이어
x = tf.placeholder(tf.float32, [None, image_size, image_size])
# 3-D shape (batch_size, width, height)을 4-D (batch_size, width, height, num_channel)로 변환
x_reshape = tf.reshape(x, [-1, image_size, image_size, 1])
# Conv 레이어
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 20], stddev=0.1))
b_conv1 = tf.Variable(tf.zeros([20]))
h_conv1 = tf.nn.tanh(tf.nn.conv2d(x_reshape, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
# Pooling 레이어
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Conv 레이어
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 20, 50], stddev=0.1))
b_conv2 = tf.Variable(tf.zeros([50]))
h_conv2 = tf.nn.tanh(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
# Pooling 레이어
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# fully-connected 레이어
W_fc1 = tf.Variable(tf.truncated_normal([50*7*7, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.zeros([500]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 50*7*7])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 출력 레이어
W_fc2 = tf.Variable(tf.truncated_normal([500, class_num], stddev=0.1))
b_fc2 = tf.Variable(tf.zeros([class_num]))
model = tf.matmul(h_fc1, W_fc2) + b_fc2

# 목표 출력
y_ = tf.placeholder(tf.int64, [None])
y_ = tf.to_int64(y_)


'''
훈련하기
'''
# 손실 함수 정의
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=model)
loss = tf.reduce_mean(cross_entropy)

# 옵티마이저 초기화 (SGD)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

# 세션 생성
sess = tf.InteractiveSession()

# 글로벌 파라미터 초기화
tf.global_variables_initializer().run()

# 학습하기
batch_num = 0
shuffle_index = [i for i in range(train_img.shape[0])]
for step in range(step_num):
  if (batch_num+1) * batch_size <= train_img.shape[0]:
    batch_xs = train_img[shuffle_index[batch_num*batch_size:(batch_num+1)*batch_size]]
    batch_ys = train_label[shuffle_index[batch_num*batch_size:(batch_num+1)*batch_size]]

    _, loss_value = sess.run([train_op, loss], feed_dict={x: batch_xs, y_: batch_ys})

    if step % 100 == 0:
      print('Step %d: loss = %.5f' % (step, loss_value))
      batch_num += 1
    else:
      batch_num = 0
      shuffle(shuffle_index)


'''
평가하기
'''
correct_prediction = tf.equal(tf.argmax(model, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

eval_accuracy = 0.0
batch_num = 0
while (batch_num+1) * batch_size <= val_img.shape[0]:
  batch_xs = val_img[batch_num * batch_size:(batch_num + 1) * batch_size]
  batch_ys = val_label[batch_num * batch_size:(batch_num + 1) * batch_size]
  step_eval_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
  eval_accuracy = eval_accuracy + step_eval_accuracy
  batch_num += 1


print('Validation accuracy: %f%%' % (eval_accuracy / batch_num * 100,))