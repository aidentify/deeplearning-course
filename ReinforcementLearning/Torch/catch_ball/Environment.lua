--[[
Reinforcement Learning / Deep Q-Learning Example
Catch ball environment

@author socurites@gmail.com
]]--

function Environment(gridSize)
    local env = {}
    local state
    local ballRow
    local ballColumn
    local basketPosition
    
    -- 새로운 게임을 시작할 때 환경을 초기화
    function env.reset()            
      -- 공의 출발 위치: (x,y) = (1, ballColumn)
      ballRow = 1
      ballColumn = math.random(1, gridSize)
      
      -- 바구니의 시작 위치: (x,y) = (basketPosition, 1)
      basketPosition = math.random(2, gridSize - 1)
    end  
    
    -- 환경의 현재 상태를 리턴.
    function env.observe()    
      local canvas = torch.Tensor(gridSize, gridSize):zero()
      canvas[ballRow][ballColumn] = 1
      canvas[gridSize][basketPosition - 1] = 1
      canvas[gridSize][basketPosition] = 1
      canvas[gridSize][basketPosition + 1] = 1

      -- 현재 상태
      state = canvas:clone()

      return state
    end
    
    -- 액션 실행하기: 에이전트(바구니) x좌표 최대 1 pixel 이동
    -- 액션 1: 왼쪽으로 이동(-1 pixel)
    -- 액션 2: 그대로 있기(0 pixel)
    -- 액션 3: 오른쪽으로 이동(+1 pixel)
    function env.act(action)
      if (action == 1) then
        move = -1
      elseif (action == 2) then
        move = 0
      else
        move = 1
      end 
      
      env._updateState(move)      
    
      return env.observe(), env._calcReward()
    end
    
    -- 액션에 따른 환경 변화 갱신
    function env._updateState(move)
      -- 공은 아래로 1칸 이동: (x,y) = (x, y+1)
      ballRow = ballRow + 1

      -- 바구니는 move 만큼 이동
      -- 10x10 canvas 화면을 벗어나는 경우 바구니 위치 그대로 두기
      basketPosition = math.min(math.max(2, basketPosition + move), gridSize - 1)  
    end
    
    -- 액션에 따른 보상/게임종료여부
    function env._calcReward()
      local reward = 0
      local gameOver = false
      if (ballRow == gridSize - 1) then                         -- 공이 canvas의 끝 부분을 통과하기 전이면
        gameOver = true
        if (math.abs(ballColumn - basketPosition) <= 1) then  -- 바구니 위에 공이 온 경우
          reward = 1
        else
          reward = -1
        end
      else
        reward = 0
      end
      
      return reward, gameOver
    end

    return env
end