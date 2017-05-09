--[[
MNIST classification in high level API

@author socurites@gmail.com
]]--

require "nn"
require "optim"
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
local path = '../data/MNIST_T7/'
local train_set = loadDataSet(path, "train_32x32.t7")
local train_label = train_set[1]
local train_img = train_set[2]
local val_set = loadDataSet(path, "test_32x32.t7")
local val_label = val_set[1]
local val_img = val_set[2]

print(#train_img)
print(#val_img)


--[[
-- 네트워크 정의
-- MLP(Multi Layer Perceptron)
]]--
-- 입력 레이어
local model = nn.Sequential();
model:add(nn.View(image_size * image_size))
-- 히든 레이어
model:add(nn.Linear(image_size * image_size, 128))
model:add(nn.ReLU())
-- 히든 레이어
model:add(nn.Linear(128, 64))
model:add(nn.ReLU())
-- 출력 레이어
model:add(nn.Linear(64, class_num))
model:add(nn.LogSoftMax())

-- 손실 함수 정의
local criterion = nn.ClassNLLCriterion()  

--[[
-- 훈련하기
]]--
local sgd_params = {
   learningRate = 0.1
}

local theta, gradTheta = model:getParameters()

-- 학습하기
for epoch = 1, epoch_num do
  local current_loss = 0
  for batch_num = 1, train_img:size(1)/batch_size, batch_size do
    local function feval(x_new)
      gradTheta:zero()
      local predictions = model:forward(train_img[{ {batch_num, batch_num + batch_size} }])
      local loss = criterion:forward(predictions, train_label[{ {batch_num, batch_num + batch_size} }])    
      local gradOutput = criterion:backward(predictions, train_label[{ {batch_num, batch_num + batch_size} }])
      model:backward(train_img[{ {batch_num, batch_num + batch_size} }], gradOutput)
      return loss, gradTheta
    end
    
    _, fs = optim.sgd(feval, theta, sgd_params)
    current_loss = current_loss + fs[1]
    print(string.format("current loss: %.2f", current_loss))
  end    
end


--[[
-- 평가하기
]]--
local correct = 0
for i=1, val_img:size(1) do
    local groundtruth = val_label[i]
    local prediction = model:forward(val_img[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(string.format("Evaluation: %.2f%s", 100 * correct / val_img:size(1), "% correct"))
