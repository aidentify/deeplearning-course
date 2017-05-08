#coding:utf-8
"""
MNIST classification in low level API

@author socurites@gmail.com
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

import mnist_data as mnist_data

'''
훈련 옵션
'''
# 미니배치 사이즈
batch_size = 100
# 학습률
learning_rate = 0.1
# 훈련 에폭
epoch_num = 10

'''
네트워크 옵션
'''
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


# 데이터 이터레이터 생성
def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32) / 255

# 훈련용 데이터 이터레이터
train_iter = mx.io.NDArrayIter(to4d(train_img), train_label, batch_size, shuffle=True)
# 평가용 데이터 이터레이터
val_iter = mx.io.NDArrayIter(to4d(val_img), val_label, batch_size)


'''
네트워크 정의
MLP(Multi Layer Perceptron)
'''
# 입력 레이어
data = mx.sym.Variable('data')
# Flat 변환 필요 없음
# data = mx.sym.Flatten(data=data)
# Conv 레이어
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# Conv 레이어
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# fully-connected 레이어
flatten = mx.sym.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# 출력 레이어
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=class_num)
model = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

# [시각화] 네트워크 시각화
shape = {"data" : (batch_size, 1, 28, 28)}
vis = mx.viz.plot_network(symbol=model, shape=shape)
vis.render('mnist-cnn')


'''
훈련하기
'''
import logging
logging.getLogger().setLevel(logging.DEBUG)

# 모듈 생성
mod = mx.mod.Module(symbol=model,
                    context=mx.gpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

# 학습하기
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate': learning_rate},
        eval_metric='acc',
        num_epoch=epoch_num,
        batch_end_callback = mx.callback.Speedometer(batch_size, 200)) # output progress for each 200 data batches


'''
평가하기
'''
valid_acc = mod.score(val_iter, ['acc'])
print('Validation accuracy: %f%%' % (valid_acc[0][1] * 100,))
