#coding:utf-8
"""
Vanilla GAN for MNIST example

@author socurites@gmail.com
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''
훈련 옵션
'''
# 미니배치 사이즈
batch_size = 100
# 학습률
learning_rate = 0.0002
# 에폭
num_epoch = 100
# 구별망 학습 스텝
num_D_steps = 3

'''
데이터 로드
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


'''
생성망 네트워크 옵션
'''
num_noise = 128

'''
구별망 네트워크 옵션
'''
__image_size = 28
input_size = __image_size * __image_size

'''
네트워크 정의(생성망)
'''
# 입력 레이어
Z = tf.placeholder(tf.float32, [None, num_noise])
# 히든 레이어
G_W1 = tf.Variable(tf.random_normal([num_noise, 256], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([256]))
G_h1 = tf.nn.relu(tf.matmul(Z, G_W1) + G_b1)
# 출력 레이어
G_W2 = tf.Variable(tf.random_normal([256, input_size], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([input_size]))
G = tf.nn.sigmoid(tf.matmul(G_h1, G_W2) + G_b2)


'''
네트워크 정의(구별망 for real)
'''
# 입력 레이어
X = tf.placeholder(tf.float32, [None, input_size])
# 히든 레이어
D_W1 = tf.Variable(tf.random_normal([input_size, 256], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([256]))
D_h1_real = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
# 출력 레이어
D_W2 = tf.Variable(tf.random_normal([256, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))
D_real = tf.nn.sigmoid(tf.matmul(D_h1_real, D_W2) + D_b2)


'''
네트워크 정의(구별망 for real)
'''
D_h1_fake = tf.nn.relu(tf.matmul(G, D_W1) + D_b1)
D_fake = tf.nn.sigmoid(tf.matmul(D_h1_fake, D_W2) + D_b2)


'''
훈련하기
'''
# 손실 함수 정의(구별망)
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))

# 손실 함수 정의(생성망)
# loss_G = tf.reduce_mean(tf.log(1 - D_fake))
loss_G = tf.reduce_mean(tf.log(D_fake))

# 옵티마이저 초기화(구별망)
# maximize loss_D = minimize -loss_D
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=[D_W1, D_b1, D_W2, D_b2])

# 옵티마이저 초기화(생성망)
# minimize (1 - D_fake) = maximize D_fake = minimize -D_fake
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=[G_W1, G_b1, G_W2, G_b2])

# 세션 생성
sess = tf.InteractiveSession()

# 글로벌 파라미터 초기화
sess.run(tf.global_variables_initializer())

# 학습하기
num_batch = int(mnist.train.num_examples/batch_size)
for epoch in range(num_epoch):
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = np.random.normal(size=(batch_size, num_noise))

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})

        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('Epoch:', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))

    if epoch % 10 == 0 or epoch == num_epoch:
        sample_size = 10
        noise = np.random.normal(size=(sample_size, num_noise))
        samples = sess.run(G, feed_dict={Z: noise})

        fig, axis = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            axis[i].set_axis_off()
            axis[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/vanilla_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
