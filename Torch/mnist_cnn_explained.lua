--[[
CNN(Convolution Neural Network) 동작 방식 설명
qtlua 설치 필요

@author socurites@gmail.com
]]--

require "nn"
require "image"
require "mnist_data"
require "optim"

--[[
-- 훈련 옵션
]]--
-- 미니배치 사이즈
batch_size = 100
-- 학습률
learning_rate = 0.1
-- 훈련 에폭
epoch_num = 10


--[[
-- 네트워크 옵션
]]--
image_size = 32
class_num = 10



--[[
-- 데이터 로드 (훈련용 / 평가용)
--]]

-- 데이터 로드
path = '../data/MNIST_T7/'
train_set = loadDataSet(path, "train_32x32.t7")
train_label = train_set[1]
train_img = train_set[2]
val_set = loadDataSet(path, "test_32x32.t7")
val_label = val_set[1]
val_img = val_set[2]

print(#train_img)
print(#val_img)


--[[
-- 네트워크 정의
-- CNN(Convolution Neural Network)
]]--
-- 입력 레이어
model = nn.Sequential();
-- Conv 레이어
model:add(nn.SpatialConvolution(1, 6, 5, 5))
model:add(nn.ReLU())
-- Pooling 레이어
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- Conv 레이어
model:add(nn.SpatialConvolution(6, 16, 5, 5))
model:add(nn.ReLU())
-- Pooling 레이어
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- fully-connected 레이어
model:add(nn.View(16*5*5))
model:add(nn.Linear(16*5*5, 120))
model:add(nn.ReLU())
-- fully-connected 레이어
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())
-- 출력 레이어
model:add(nn.Linear(84, class_num))
model:add(nn.LogSoftMax())

-- 손실 함수 정의
criterion = nn.ClassNLLCriterion()


--[[
-- CNN 레이어 이해하기
-- CNN(Convolution Neural Network)
]]--
-- 8번째 이미지의 레이블
 =train_label[3]
-- 5

-- 8번째 이미지 출력
four = train_img[3]
image.display(four)

-- 모델의 레이어 출력
=model
--[[
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> output]
  (1): nn.SpatialConvolution(1 -> 6, 5x5)
  (2): nn.ReLU
  (3): nn.SpatialMaxPooling(2x2, 2,2)
  (4): nn.SpatialConvolution(6 -> 16, 5x5)
  (5): nn.ReLU
  (6): nn.SpatialMaxPooling(2x2, 2,2)
  (7): nn.View(400)
  (8): nn.Linear(400 -> 120)
  (9): nn.ReLU
  (10): nn.Linear(120 -> 84)
  (11): nn.ReLU
  (12): nn.Linear(84 -> 10)
  (13): nn.LogSoftMax
]]--

-- 하나의 이미지에 대한 피드 포워드
out = model:forward(four)

=out
--[[
-2.2484
-2.2807
-2.3750
-2.1926
-2.3383
-2.4132
-2.3736
-2.2329
-2.3560
-2.2405
[torch.DoubleTensor of size 10]
]]--

-- 첫번째 컨볼루션 레이어
= model:get(1)
-- nn.SpatialConvolution(1 -> 6, 5x5)

-- 첫번째 컨볼루션 레이어의 가중치(커널 필터) 출력
image.display { image = model:get(1).weight, legend = 'k(1)', zoom = 18, padding = 2 }

-- 첫번째 컨볼루션 레이어의 결과값 출력
image.display { image = model:get(1).output, legend = 'y(1)', scaleeach=true }

-- 두번째 활성화 함수(ReLU) 레이어
= model:get(2)
-- nn.ReLU

-- 두번째 활성화 함수(ReLU) 레이어의 결과값 출력
image.display { image = model:get(2).output, legend = 'relu(y(1))', scaleeach=true }

-- 세번째 풀링 레이어
= model:get(3)
-- nn.SpatialMaxPooling(2x2, 2,2)

-- 세번째 풀링 레이어의 결과값 출력
image.display { image = model:get(3).output, legend = 'pool(relu(y(1)))', scaleeach=true }


-- 나머지 레이어의 결과값 출력
image.display { image = model:get(4).output, legend = 'y(2)', scaleeach=true }
image.display { image = model:get(5).output, legend = 'relu(y(2))', scaleeach=true }
image.display { image = model:get(6).output, legend = 'pool(relu(y(2)))', scaleeach=true }