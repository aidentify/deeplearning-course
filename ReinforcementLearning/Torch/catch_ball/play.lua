--[[
Reinforcement Learning / Deep Q-Learning Example
Catch ball play runnable by qlua

@author socurites@gmail.com
]]--

require 'nn'
require 'Environment'
require 'QlearningAgent'
require 'image'

--[[
-- 상태(state) 정의
]]--
local gridSize = 10
local numAction = 3
local renderSize = 512
local maxGames = 100
local backgroundColour = { 203, 158, 119 } -- B,G,R

--[[
-- greedy action 파라미터 정의
-- no exploration
]]--
local epsilon = 0.0


local displayImage = torch.Tensor(3, renderSize, renderSize)
local function processImage(state)
  local display = image.scale(state, renderSize, renderSize, 'simple')
  for i = 1, displayImage:size(1) do
    displayImage[i]:copy(display)
    displayImage[i][torch.le(display, 0)] = backgroundColour[i]
  end
  return displayImage
end

local function drawState(image, painter)
    painter.image = image
    local size = painter.window.size:totable()
    painter.refresh(size.width, size.height)
end

local function sleep(n)
    os.execute("sleep " .. tonumber(n))
end


--[[
-- 환경 생성
]]--
local catchEnvironment = Environment(gridSize)

--[[
-- Qlearning 에이전트 생성
]]--
local qlearningAgent = QlearningAgent(gridSize, numAction, epsilon, 0.0, 0.0)
qlearningAgent.init()
qlearningAgent.load("TorchQLearningModel.t7")

--[[
-- 게임 화면 렌더링
]]--
catchEnvironment.reset()
local currentState = catchEnvironment.observe()

local display = processImage(currentState)
local painter = image.display(display)
painter.window.windowTitle = 'TorchPlaysCatch'
drawState(display, painter)

--[[
-- 게임 실행
]]--
local numberOfGames = 0
while numberOfGames < maxGames do
    local isGameOver = false
    catchEnvironment.reset()
    local currentState = catchEnvironment.observe()
    drawState(processImage(currentState), painter)

    while (isGameOver ~= true) do          
      local action = qlearningAgent.chooseAction(currentState)
      local nextState, reward, gameOver = catchEnvironment.act(action)
      currentState = nextState
      isGameOver = gameOver
      drawState(processImage(currentState), painter)
      sleep(0.05)
    end
    collectgarbage()
end


local env = Environment(gridSize)

