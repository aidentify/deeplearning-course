--[[
Reinforcement Learning / Deep Q-Learning Example
TicTacToe play 
Qlearning agent first
@author socurites@gmail.com
]]--

require 'nn'
require 'TicTacToeEnvironment'
require 'image'
require 'TicTacToeUtil'
require 'TicTacToeQLearningAgent'
require 'cutorch'
require 'cunn'
require 'cudnn'


local cmd = torch.CmdLine()
cmd:text('Training options')
--cmd:option('-epoch', 60000, 'The epoch of pre-trained model')
cmd:option('-epoch', 10, 'The epoch of pre-trained model')
cmd:option('-gridSize', 3, 'The size of the grid that the agent is going to play the game on.')
cmd:option('-numAction', 9, 'The number of actions.')

local opt = cmd:parse(arg)
local numAction = opt.numAction
local epoch = opt.epoch
local gridSize = opt.gridSize
local epsilon = 0.0



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
agentO.load("tictactoc-" .. epoch .. "-O.t7")


--[[
for i = 1, 100 do
  env.reset()
  local gameOver = false
  local reward = 0
  
  local currState, nextState
  local action
  
  while (gameOver ~= true) do    
    
    io.write("choose action (1 ~ 9): ")
    io.flush()
    action = tonumber(io.read())
    
    -- Update Enviroiment
    nextState, reward, gameOver = env.act(action, 1)
    
    util.printBoard( nextState )
    print(nextState)
    
    if ( gameOver == true ) then
      if ( reward == 1 ) then
        print "You Win"
      else
        print "Draw"
      end
      break
    end
    
    
    
    
    -- Later        
    currState = env.observe():clone()
    action = agentO.chooseAction(currState)
    print('[agentO] : ' .. action)
    
    -- Update Enviroiment
    nextState, reward, gameOver = env.act(action, agentO.stone())
    
    util.printBoard( nextState )
    print(nextState)
    
    if ( gameOver == true ) then
      if ( reward == 1 ) then
        print "You Lose"
      else
        print "Draw"
      end
      break
    end        

  end
end
]]--


for i = 1, 100 do
  env.reset()
  local gameOver = false
  local reward = 0
  
  local currState, nextState
  local action
  
  while (gameOver ~= true) do
    currState = env.observe():clone()
    --print ( currState )
    
    -- First        
    action = agentO.chooseAction(currState)
    print('[agentO] : ' .. action)
    
    -- Update Enviroiment
    nextState, reward, gameOver = env.act(action, agentO.stone())
    
    util.printBoard( nextState )
    print(nextState)
    
    if ( gameOver == true ) then
      if ( reward == 1 ) then
        print "You Lose"
      else
        print "Draw"
      end
      break
    end
    
    
    io.write("choose action (1 ~ 9): ")
    io.flush()
    action = tonumber(io.read())
    
    -- Update Enviroiment
    nextState, reward, gameOver = env.act(action, 1)
    
    util.printBoard( nextState )
    print(nextState)
    
    if ( gameOver == true ) then
      if ( reward == 1 ) then
        print "You Win"
      else
        print "Draw"
      end
      break
    end
  end
end
