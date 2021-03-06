# deeplearning-course
"인공지능, 머신러닝, 딥러닝 입문과정"을 위한 실습 예제로 아래와 딥러닝 프레임워크를 이용하여 예제를 구현한다.

## 실습 딥러닝 프레임워크
* Torch
* MXNet
* TensorFlow

## 실습 예제
* MNIST 필기체 분류
* 강화학습 - 공받기 게임
* 강화학습 - 틱택토(TicTacToe)
* GAN - 정규 분포 생성
* RNN/LSTM - 발라드 가사 생성


## data
MNIST 필기체 분류 학습을 위한 훈련 데이터셋과 평가를 위한 평가 데이터셋으로 구성된다. 
* MNIST/
  * MNIST 필기체 바이트 데이터
  * 이미지 shape: 1 x 28 x 28
* MNIST_T7/
  * Torch 텐서(Tensor) 포맷에 맞게 직렬화한 데이터
  * 이미지 shape: 1 x 32 x 32


## Torch
MNIST 필기체 분류 학습을 Torch를 이용하여 구현한 예제
* mnist_classify_basic.lua
  * MLP(Multi Layer Perceptron)을 이용하여 구현한 예제
* mnist_classify_high.lua
  * 위의 예제를 optim 패키지를 이용하여 구현한 예제
* mnist_cnn_explained
  * CNN 내부 처리 과정을 설명하는 예제
* mnist_classify_cnn.lua
  * CNN(Convolution Neural Network)을 이용하여 구현한 예제


## MXNet
MNIST 필기체 분류 학습을 MXNet를 이용하여 구현한 예제
* mnist_classify_basic.lua
  * MLP(Multi Layer Perceptron)을 이용하여 구현한 예제
* mnist_classify_high.lua
  * 위의 예제를 Module.fit()을 이용하여 구현한 예제
* mnist_classify_cnn.lua
  * CNN(Convolution Neural Network)을 이용하여 구현한 예제


## TensorFlow
MNIST 필기체 분류 학습을 TensorFlow를 이용하여 구현한 예제
* mnist_classify_high.lua
  * MLP(Multi Layer Perceptron)을 이용하여 구현한 예제
* mnist_classify_vis.lua
  * 위의 예제에 텐서보드(TensorBoard) 시각화를 위해 tf.summary 메트릭을 적용한 예제
* mnist_classify_cnn.lua
  * CNN(Convolution Neural Network)을 이용하여 구현한 예제


## ReinforcementLearning
강화학습을 이용하여 공받기게임과 TicTacToe를 구현한 예제


## GAN
GAN을 이용하여 정규 분포 생성을 구현한 예제
