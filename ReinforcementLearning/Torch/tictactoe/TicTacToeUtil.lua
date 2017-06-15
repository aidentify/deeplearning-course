--[[
Reinforcement Learning / Deep Q-Learning Example
tictactoe utility functions

@author socurites@gmail.com
]]--

function TicTacToeUtil()
  local util = {}
  
  -- epsilon-greedy 액션 선택을 위한 랜덤값 리턴
  function util.randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
  end
  
  -- -1/0/1로 표현된 내부 상태를 O/ / X로 표현하여 출력
  function util.printBoard(state)
    for i = 1, 3 do
      for j = 1, 3 do
        if ( j== 1 ) then io.write('|') end
        
        if ( state[i][j] == -1 ) then
          io.write('O|')
        elseif ( state[i][j] == 1 ) then
          io.write('X|')
        else
          io.write(' |')
        end
      end
      print('')
    end
  end
  
  -- 액션 숫자를 [x][y] 좌표로 변환하여 리턴
  -- [1 2 3
  --  4 5 6
  --  7 8 9]
  -- @param num int
  -- @return [x][y] coordinate
  function util.coord(num)
    if ( num <= 3 ) then
        return {1, num}
    elseif ( num <= 6 ) then
        return {2, num-3}
    else
        return {3, num-6}
    end
  end
  
  -- 실행가능한 액션인지 확인
  -- 이미 수가 놓인 곳에는 중복해서 수를 놓지 못하도록 함
  function util.isActionable(canvas, action)
    local coord = util.coord(action)
    
    if ( canvas[coord[1]][coord[2]] == 0 ) then
        return true
    end
    return false
  end 

  
  -- Check winning status
  function util.checkWinState(canvas, stone)
    -- horizontal
    for i = 1, 3 do
        if ( canvas[i][1] == stone and canvas[i][2] == stone and canvas[i][3] == stone ) then
            return true
        end
    end
    -- vertical
    for i = 1, 3 do
        if ( canvas[1][i] == stone and canvas[2][i] == stone and canvas[3][i] == stone ) then
            return true
        end
    end
    -- diagonal
    if ( canvas[1][1] == stone and canvas[2][2] == stone and canvas[3][3] == stone ) then
        return true
    end
    if ( canvas[1][3] == stone and canvas[2][2] == stone and canvas[3][1] == stone ) then
        return true
    end
    return false
  end
  
  function util.isAllMarked(canvas)
    for i= 1, 3 do
      for j = 1, 3 do
        if ( canvas[i][j] == 0 ) then
          return false
        end
      end
    end
    return true     
  end

  return util
end