--[[
Reinforcement Learning / Deep Q-Learning Example
tictactoe

@author socurites@gmail.com
]]--
require 'nn'
require 'optim'
require 'TicTacToeMemory'
require 'TicTacToeEnvironment'
require 'TicTacToeRandomAgent'
require 'TicTacToeQLearningAgent'
require 'TicTacToeUtil'
require 'cutorch'
require 'cunn'
require 'cudnn'

local cmd = torch.CmdLine()
cmd:text('훈련 옵션')
cmd:option('-epsilon'           , 0.9         , 'epsilon-greedy 액션 선택 확률. (0 to 1)')
cmd:option('-epsilonDecay'      , 0.9999     , 'epsilon 감쇠 계수 (0 to 1)')
cmd:option('-epsilonMinVal'     , 0.001       , 'epsion 최소값. (0 to 1)')
cmd:option('-numAction'        , 9           , '선택가능학 액션 수.')
cmd:option('-epoch'             , 300000      , '훈련 에폭 수.')
cmd:option('-maxMemory'         , 300000      , 'replay memory 사이즈.')
cmd:option('-batchSize'         , 100         , 'mini-batch 사이즈.')
cmd:option('-gridSize'          , 3           , '그리드 사이즈.')
cmd:option('-discount'          , 0.9         , 'discount rate')
cmd:option('-learningRate'      , 1e-3        , '학습률. (0 to 1)')
cmd:option('-learningRateDecay' , 1e-9        , '학습률 감쇠 게수. (0 to 1)')
cmd:option('-weightDecay'       , 0.0         , '가중치 감쇠 계수. (0 to 1)')
cmd:option('-momentum'          , 0.9         , '모멘텀')
cmd:option('-savePrefix'        , 'tictactoc-', '모델 저장 파일명 prefix.')

local opt = cmd:parse(arg)

local epsilon       = opt.epsilon
local epsilonDecay  = opt.epsilonDecay
local epsilonMinVal = opt.epsilonMinVal
local numAction    = opt.numAction
local epoch         = opt.epoch
local maxMemory     = opt.maxMemory
local batchSize     = opt.batchSize
local gridSize      = opt.gridSize
local numStates     = gridSize * gridSize
local discount      = opt.discount

math.randomseed(os.time())
local util = TicTacToeUtil()

--[[
-- 환경 생성
]]--
local env = TicTacToeEnvironment(gridSize)


--[[
-- QlearningAgent 생성
]]--
-- 플레이어 O 에이전트
local agentO = TicTacToeQLearningAgent(gridSize, numAction, -1, epsilon)
agentO.init()
-- 플레이어 X 에이전트
local agentX = TicTacToeQLearningAgent(gridSize, numAction, 1, epsilon)
agentX.init()


--[[
-- Replay Memory 생성
]]--
-- 플레이어 O 메모리
memoryO = TicTacToeMemory(maxMemory, gridSize, numAction, discount)
-- 플레이어 X 메모리
memoryX = TicTacToeMemory(maxMemory, gridSize, numAction, discount)


--[[
-- 훈련하기
]]--
-- 하이퍼 파라미터 정의 for O
local sgdParamsO = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true,
    beta1 = 0.9,
    beta2 = 0.999
}

-- 하이퍼 파라미터 정의 for X
local sgdParamsX = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true,
    beta1 = 0.9,
    beta2 = 0.999
}

local winOCount = 0
local winXCount = 0
local drawCount = 0
for i = 1, epoch do  
  local errO = 0
  local errX = 0
  local gameOver = false
  local reward = 0
  
  env.reset()
  
  local currState
  local nextState
  local action
  
  local transition
  gameStatus = {
    gameOver = false,
    gameResult = nil
  }
  episode = {}
  while (gameOver ~= true) do
    -- [에이전트 O] 현재 상태 관측
    currState = env.observe()  
    
    -- [에이전트 O] 액션 선택하기
    action = agentO.chooseAction(currState)
    
    -- [에이전트 O] 액션(action) 실행 및
    -- 환경(environment)으로부터
    -- 다음 상태(state) 관측
    -- 보상(reward) 획득
    nextState, reward, gameOver = env.act(action, agentO.stone())    
    
    -- 게임 시퀀스 저장
    transition = {
      inputState = currState,
      action = action,
      reward = 0,
      nextState = nextState,
      gameOver = false
    }
    table.insert(episode, transition)
    
    -- 게임 상태 갱신
    if ( gameOver == true ) then 
      gameStatus.gameOver = true
      if ( reward >= 1 ) then        
        gameStatus.gameResult = 'O Win' 
        winOCount = winOCount + 1
      elseif ( reward <= -1 ) then
        gameStatus.gameResult = 'X Win'
        winXCount = winXCount + 1
      else
        gameStatus.gameResult = 'Draw'   
        drawCount = drawCount + 1
      end    
      
      break
    end

    -- [에이전트 X] 현재 상태 관측
    currState = env.observe()
    
    -- [에이전트 X] 액션 선택하기
    action = agentX.chooseAction(currState)
    
    -- [에이전트 X] 액션(action) 실행 및
    -- 환경(environment)으로부터
    -- 다음 상태(state) 관측
    -- 보상(reward) 획득
    nextState, reward, gameOver = env.act(action, agentX.stone())
    
    -- 게임 시퀀스 저장
    transition = {
      inputState = currState,
      action = action,
      reward = 0,
      nextState = nextState,
      gameOver = false
    }
    table.insert(episode, transition)
            
    -- 게임 상태 갱신
    if ( gameOver == true ) then 
      gameStatus.gameOver = true
      if ( reward >= 1 ) then
        gameStatus.gameResult = 'X Win'        
        winXCount = winXCount + 1
      elseif ( reward <= -1 ) then
        gameStatus.gameResult = 'O Win'
        winOCount = winOCount + 1
      else
        gameStatus.gameResult = 'Draw'
        drawCount = drawCount + 1
      end  
      
      break
    end
  end

  
  for i=1, #episode, 2 do    
    local transition = episode[i]
    if ( i+1 <= #episode ) then
      transition.nextState = episode[i+1].nextState  
    end
    
    if ( i >= #episode-1 ) then  
      transition.gameOver = true
      if ( gameStatus.gameResult == 'O Win' ) then
        transition.reward = 1
      elseif ( gameStatus.gameResult == 'X Win' ) then
        transition.reward = -1
      else
        transition.reward = 0.5
      end
    end
      
    memoryO.remember(transition)
  end
  
  
  for i=2, #episode, 2 do    
    local transition = episode[i]        
    if ( i+1 <= #episode ) then
      transition.nextState = episode[i+1].nextState  
    end
    
    if ( i >= #episode-1 ) then  
      transition.gameOver = true
      if ( gameStatus.gameResult == 'X Win' ) then
        transition.reward = 1
      elseif ( gameStatus.gameResult == 'O Win' ) then
        transition.reward = -1
      else
        transition.reward = 0.5
      end
    end
        
    memoryX.remember(transition)
  end
  
    -- [에이전트 O] 학습 mini-batch 선택하기
    local inputs, targets = memoryO.getBatch(batchSize, agentO)    
    
    -- [에이전트 O] 에이전트 훈련
    errO = errO + agentO.train(inputs, targets, sgdParamsO)
    
    -- [에이전트 X] 학습 mini-batch 선택하기
    local inputs, targets = memoryX.getBatch(batchSize, agentX)    
    
    -- [에이전트 X] 에이전트 훈련
    errX = errX + agentX.train(inputs, targets, sgdParamsX)

    -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
    if (epsilon > epsilonMinVal) then
        epsilon = epsilon * epsilonDecay
    end
  
  if ( i % 100 == 0 ) then
    print(string.format("Epoch %d %.5f: [%s] [WinO Rate = %.2f WinX Rate = %.2f Draw Rate = %.2f] err = %.5f : err = %.5f : WinO count %d : WinX count %d : Draw Count %d", i, epsilon, gameStatus.gameResult, winOCount / i * 100, winXCount / i * 100, drawCount / i * 100, errO, errX, winOCount, winXCount, drawCount))      
  end
  
  if ( i == 10 ) then
    torch.save(opt.savePrefix .. i .. '-O' .. '.t7', agentO.getModel())
    torch.save(opt.savePrefix .. i .. '-X' .. '.t7', agentX.getModel())
  end

  if ( i > 0 and i % 3000 == 0 ) then
    torch.save(opt.savePrefix .. i .. '-O' .. '.t7', agentO.getModel())
    torch.save(opt.savePrefix .. i .. '-X' .. '.t7', agentX.getModel())
  end
end
