--[[
Reinforcement Learning / Deep Q-Learning Example
Catch ball game

@author socurites@gmail.com
]]--

require 'nn'
require 'optim'
require 'Environment'
require 'QlearningAgent'
require 'ReplayMemory'

--[[
-- 훈련 옵션
]]--
-- 미니배치 사이즈
local batch_size = 100
-- 학습률
local learning_rate = 0.1
-- 훈련 에폭
local epoch_num = 1000
-- 학습률 감쇄
local learning_rate_decay = 1e-9
-- 가중치 감쇄
local weight_decay = 0.0


--[[
-- 상태(state) 정의
]]--
local gridSize = 10
local numAction = 3

--[[
-- epsilon-greedy action 파라미터 정의
]]--
local epsilon = 1
local epsilonDecay = 0.999
local epsilonMinimumValue = 0.001

--[[
-- replay memory 파라미터 정의
]]--
local maxMemory = 500
local discount = 0.9  -- discount factor

math.randomseed(os.time())

--[[
-- 환경 생성
]]--
local catchEnvironment = Environment(gridSize)

--[[
-- Qlearning 에이전트 생성
]]--
local qlearningAgent = QlearningAgent(gridSize, numAction, epsilon, epsilonDecay, epsilonMinimumValue)
qlearningAgent.init()


--[[
-- Replay Memory 생성
]]--
local replayMemory = ReplayMemory(maxMemory, gridSize, numAction, discount)


--[[
-- 훈련하기
]]--
-- 하이퍼 파라미터 정의
sgdParams = {
    learningRate = learning_rate,
    learningRateDecay = learning_rate_decay,
    weightDecay = weight_decay,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

local winCount = 0
for i = 1, epoch_num do
    local err = 0
    catchEnvironment.reset()
    local isGameOver = false

    -- 현재 상태 관측
    local currentState = catchEnvironment:observe()
    --print(currentState)

    while (isGameOver ~= true) do
      -- 액션 선택하기
      -- epsilon-greedy exploration
      local action = qlearningAgent.chooseAction(currentState)
      
      -- 액션(action) 실행 및
      -- 환경(environment)으로부터
      -- 다음 상태(state) 관측
      -- 보상(reward) 획득
      nextState, reward, gameOver = catchEnvironment.act(action)
      --print(nextState)
      
      if (reward == 1) then winCount = winCount + 1 end

      -- 새로운 경험 저장
      replayMemory.remember(currentState, action, reward, nextState, gameOver)
      
      --학습 mini-batch 선택하기
      local inputs, targets = replayMemory.getBatch(batch_size, qlearningAgent)
      
      -- 에이전트 훈련
      err = err + qlearningAgent.train(inputs, targets, sgdParams)
        
      -- 상태 갱신
      currentState = nextState
      isGameOver = gameOver
    end
    print(string.format("Epoch %d : err = %f : Win count %d ", i, err, winCount))
end


--[[
-- 모델 저장하기
]]--
local savePath = 'TorchQLearningModel.t7'
torch.save(savePath, qlearningAgent.getModel())
print("Model saved to " .. savePath)





