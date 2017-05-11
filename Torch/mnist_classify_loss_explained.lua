--[[
MNIST classification in low level API

@author socurites@gmail.com
]]--

require "nn"
require "mnist_data"


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
-- MLP(Multi Layer Perceptron)
]]--
-- 입력 레이어
model = nn.Sequential();
model:add(nn.View(image_size * image_size))
-- 히든 레이어
model:add(nn.Linear(image_size * image_size, 128))
model:add(nn.ReLU())
-- 히든 레이어
model:add(nn.Linear(128, 64))
model:add(nn.ReLU())
-- 출력 레이어
model:add(nn.Linear(64, class_num))

-- 손실 함수 정의
criterion = nn.CrossEntropyCriterion()  


--[[
-- 훈련하기
]]--
theta, gradTheta = model:getParameters()

-- 3번째 이미지의 레이블
=train_label[3]
-- 5

-- 하나의 이미지에 대한 피드 포워드
prediction = model:forward(train_img[3])

=prediction

-- 하나의 이미지에 대한 손실
loss = criterion:forward(prediction, train_label[3])

=loss

sum_exp_pred = 0.
for i=1,10 do
  sum_exp_pred = sum_exp_pred + torch.exp(prediction[i])
end


=-torch.log(torch.exp(prediction[5]) / sum_exp_pred)
-- is equal to loss


