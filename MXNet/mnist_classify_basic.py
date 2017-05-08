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


# 데이터 이터레이터 생성
def to4d(img):
    return img.reshape(img.shape[0], 1, image_size, image_size).astype(np.float32) / 255

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
# 4-D shape (batch_size, num_channel, width, height)을 2-D (batch_size, num_channel*width*height)로 변환
data = mx.sym.Flatten(data=data)
# 히든 레이어
fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
# 히든 레이어
fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")
# 출력 레이어
fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=class_num)
model = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# [시각화] 네트워크 시각화
shape = {"data" : (batch_size, 1, image_size, image_size)}
vis = mx.viz.plot_network(symbol=model, shape=shape)
vis.render('mnist-mlp-basic')


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

# 데이터 바인드
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

# 파라미터 초기화 (Xavier)
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

# 옵티마이저 초기화 (SGD)
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', learning_rate),))

# 훈련 메트릭 생성
metric = mx.metric.create('acc')

# 학습하기
for i in range(epoch_num):
    print("#%d " % i, end='')
    train_iter.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)         # forward pass
        mod.update_metric(metric, batch.label)
        mod.backward()                            # back propagation
        mod.update()                              # update parameters
    print(metric.get())

print('Training accuracy: %f%%' % (metric.get()[1] * 100,))


'''
평가하기
'''
valid_acc = mod.score(val_iter, ['acc'])
print('Validation accuracy: %f%%' % (valid_acc[0][1] * 100,))
