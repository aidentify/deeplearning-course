--[[
Reinforcement Learning / Deep Q-Learning Example
TicTacToe environment

@author socurites@gmail.com
]]--

function TicTacToeEnvironment(gridSize)
  local env = {}
  local state
  local currState
  local nextState
  local util = TicTacToeUtil()
  
  -- 새로운 게임을 시작할 때 환경을 초기화
  function env.reset()
    state = torch.Tensor(gridSize, gridSize):zero()
  end
      
  -- 환경의 현재 상태를 리턴.
  function env.observe()        
    return state:clone():cuda()
  end
  
  -- 액션 실행하기: action에 해당하는 좌표에 stone 놓기
  function env.act(action, stone)  
    local coord = util.coord(action)
    state[coord[1]][coord[2]] = stone
    
    local reward = 0
    local gameOver = false
    local winningCond = util.checkWinState(state, stone)    
    if ( winningCond == true ) then
      reward = 1
      gameOver = true
    elseif ( util.isAllMarked(state) ) then
      reward = 0.5
      gameOver = true        
    end
    
    return env.observe(), reward, gameOver
  end

  return env
end
