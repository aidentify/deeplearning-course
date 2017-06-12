--[[
Reinforcement Learning / Deep Q-Learning Example
Catch ball game

@author socurites@gmail.com
]]--

require 'nn'
require 'optim'

--[[
-- 훈련 옵션
]]--
-- 미니배치 사이즈
batch_size = 100
-- 학습률
learning_rate = 0.1
-- 훈련 에폭
epoch_num = 10
-- 학습률 감쇄
learning_rate_decay = 1e-9,
-- 가중치 감쇄
weight_decay = 0.0


--[[
-- 상태(state) 정의
]]--
gridSize = 10
numAction = 3

-- 공의 출발 위치: (x,y) = (1, ballColumn)
ballRow = 1
ballColumn = math.random(1, gridSize)

-- 바구니의 시작 위치: (x,y) = (basketPosition, 1)
basketPosition = math.random(2, gridSize - 1)

canvas = torch.Tensor(gridSize, gridSize):zero()
canvas[ballRow][ballColumn] = 1
canvas[gridSize][basketPosition - 1] = 1
canvas[gridSize][basketPosition] = 1
canvas[gridSize][basketPosition + 1] = 1

-- 현재 상태
currentState = canvas:clone()

print(currentState)


--[[
-- DQN 네트워크 정의
-- MLP(Multi Layer Perceptron)
]]--
model = nn.Sequential()
-- 입력 레이어
model:add(nn.View(gridSize * gridSize))
model:add(nn.Linear(gridSize * gridSize, 100))
model:add(nn.ReLU())
-- 히든 레이어
model:add(nn.Linear(100, 100))
model:add(nn.ReLU())
-- 출력 레이어
model:add(nn.Linear(100, numAction))

-- 손실 함수 정의
criterion = nn.MSECriterion()


--[[
-- 액션 선택하기
-- epsilon-greedy exploration
]]--
epsilon = 1
epsilonDecay = 0.999
epsilonMinimumValue = 0.001
actionBy = ''

function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end

if (randf(0, 1) <= epsilon) then
    action = math.random(1, numAction)
    actionBy = "exploration"
else
    q = model:forward(currentState)
    _, index = torch.max(q, 1)
    action = index[1]
    actionBy = "exploitation"
end

print(actionBy .. ": " .. action)

-- Decay the epsilon
if (epsilon > epsilonMinimumValue) then
    epsilon = epsilon * epsilonDecay
end


--[[
-- 액션 실행하기: 에이전트(바구니) x좌표 최대 1 pixel 이동
-- 액션 1: 왼쪽으로 이동(-1 pixel)
-- 액션 2: 그대로 있기(0 pixel)
-- 액션 3: 오른쪽으로 이동(+1 pixel)
]]--

if (action == 1) then
    move = -1
elseif (action == 2) then
    move = 0
else
    move = 1
end


--[[
-- 환경(environment)으로부터
-- 다음 상태(state) 관측
-- 보상(reward) 획득
]]--

-- 공은 아래로 1칸 이동: (x,y) = (x, y+1)
ballRow = ballRow + 1

-- 바구니는 move 만큼 이동
-- 10x10 canvas 화면을 벗어나는 경우 바구니 위치 그대로 두기
basketPosition = math.min(math.max(2, basketPosition + move), gridSize - 1)

canvas = torch.Tensor(gridSize, gridSize):zero()
canvas[ballRow][ballColumn] = 1
canvas[gridSize][basketPosition - 1] = 1
canvas[gridSize][basketPosition] = 1
canvas[gridSize][basketPosition + 1] = 1

-- 다음 상태 관측
nextState = canvas:clone()

print(nextState)

-- 보상 획득
reward = 0
gameOver = false
winCount = 0
if (ballRow == gridSize - 1) then                         -- 공이 canvas의 끝 부분을 통과하기 전이면
    gameOver = true
    if (math.abs(ballColumn - basketPosition) <= 1) then  -- 바구니 위에 공이 온 경우
         reward = 1
         winCount = winCount + 1
    else
        reward = -1
    end
else
    reward = 0
end


--[[
-- replay memory에 경험(experience) 저장하기
]]--

-- replay memory 생성
memory = {}

-- replay memory에 경험(experience) 저장하기
table.insert(memory, {
    inputState = currentState,
    action = action,
    reward = reward,
    nextState = nextState,
    gameOver = gameOver
});


--[[
-- 훈련 데이터셋 미니배치 선택하기
]]--

-- 미니배치 경험 랜덤으로 선택하기
memoryLength = #memory
chosenBatchSize = math.min(batch_size, memoryLength)
inputs = torch.Tensor(chosenBatchSize, gridSize, gridSize):zero()
targets = torch.Tensor(chosenBatchSize, numAction):zero()

-- 1개의 경험에 대해 훈련할 input/target 데이터 생성
randomIndex = math.random(1, memoryLength)
memoryInput = memory[randomIndex]

-- Q-value 계산
if (memoryInput.gameOver) then
  target[memoryInput.action] = memoryInput.reward
else
  discount = 0.9  -- discount factor
   
  -- 상태 s에서 실행가능한 모든 액션 a에 대해 Q(s,a) 계산
  target = model:forward(memoryInput.inputState):clone()
   
  -- max_a' Q(s',a')계산
  nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
  
  -- 상태 s에 대해서 목표값(target)을  r + γmax a’ Q(s’, a’)로 설정
  target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
end

-- 미니배치 1번째 아이템 추가
inputs[1] = memoryInput.inputState
targets[1] = target


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

-- 학습하기
current_loss = 0.0
theta, gradTheta = model:getParameters()
function feval(x_new)
    gradTheta:zero()
    predictions = model:forward(inputs)
    loss = criterion:forward(predictions, targets)
    gradOutput = criterion:backward(predictions, targets)
    model:backward(inputs, gradOutput)
    return loss, gradTheta
end

_, fs = optim.sgd(feval, theta, sgdParams)
current_loss = current_loss + fs[1]
print(string.format("current loss: %.2f", current_loss))