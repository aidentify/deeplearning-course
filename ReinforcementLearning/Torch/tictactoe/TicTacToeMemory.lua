--[[
Reinforcement Learning / Deep Q-Learning Example
Replay Memory

@author socurites@gmail.com
]]--
function TicTacToeMemory(maxMemory, gridSize, numAction, discount)
    local memory = {}
    
    -- 새로운 경험 (s,a,r,s') 기억
    function memory.remember(memoryInput)
      table.insert(memory, memoryInput)
      if (#memory > maxMemory) then
          table.remove(memory, 1)
      end
    end

  -- 학습 mini-batch 선택
  function memory.getBatch(batch_size, qlearningAgent)
    local memoryLength = #memory
    local chosenBatchSize = math.min(batch_size, memoryLength)
    local inputs = torch.Tensor(chosenBatchSize, gridSize, gridSize):cuda():zero()
    local targets = torch.Tensor(chosenBatchSize, numAction):cuda():zero()
    
    for i = 1, chosenBatchSize do
      local randomIndex = math.random(1, memoryLength)
      local memoryInput = memory[randomIndex]
      

      -- Q-value 계산
      -- 상태 s에서 실행가능한 모든 액션 a에 대해 Q(s,a) 계산
      local target = qlearningAgent.forward(memoryInput.inputState):clone()
   
      -- max_a' Q(s',a')계산
      nextStateMaxQ = torch.max(qlearningAgent.forward(memoryInput.nextState), 1)[1]
        
      if (memoryInput.gameOver) then
        target[memoryInput.action] = memoryInput.reward
      else
        -- 상태 s에 대해서 목표값(target)을  r + γmax a’ Q(s’, a’)로 설정
        target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
      end      
      inputs[i] = memoryInput.inputState
      targets[i] = target
    end
    
    if ( inputs:size(1) == 1 ) then
        inputs_1 = inputs:view(-1)
    else
        inputs_1 = inputs
    end
    _, index = torch.max(targets, 2)
    targets_1 = index:view(-1)    
    
    return inputs_1, targets_1
  end
    
  return memory
end
