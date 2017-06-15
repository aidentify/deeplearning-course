require 'nn'
require 'TicTacToeEnvironment'
require 'image'
require 'TicTacToeUtil'
require 'TicTacToeQLearningAgent'

local cmd = torch.CmdLine()
cmd:text('Training options')
--cmd:option('-epoch', 60000, 'The epoch of pre-trained model')
cmd:option('-epoch', 213000, 'The epoch of pre-trained model')
cmd:option('-gridSize', 3, 'The size of the grid that the agent is going to play the game on.')
cmd:option('-numActions', 9, 'The number of actions.')

local opt = cmd:parse(arg)
local numActions = opt.numActions
local epoch = opt.epoch
local gridSize = opt.gridSize
local epsilon = 0.0


math.randomseed(os.time())

local util = TicTacToeUtil()

local model = torch.load("tictactoc-" .. epoch .. "-X.t7")

local env = TicTacToeEnvironment(gridSize)

local agentO = TicTacToeQLearningAgent(numActions, model, -1)
local agentX = TicTacToeQLearningAgent(numActions, model, 1)


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
    nextState, reward, gameOver = env.act(action, -1)
        
    if ( gameOver == true ) then
      if ( reward == 1 ) then
        print "You Win"
      else
        print "Draw"
      end
      break
    end
    
    currState = env.observe():clone()
    print(currState)
    
    action = agentX.chooseAction(currState, epsilon)
    print('[agentX] : ' .. action)
    
    -- Update Enviroiment
    nextState, reward, gameOver = env.act(action, agentX.stone())
    
    print ( nextState )
    
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
