--[[
Reinforcement Learning / Deep Q-Learning Example
Qlearning agent

@author socurites@gmail.com
]]--

require 'TicTacToeUtil'

function TicTacToeQLearningAgent(gridSize, numActions, stone, epsilon)
  local agent = {}
  local model
  local criterion
  local util = TicTacToeUtil()
  
  -- Qlearning agent 초기화
  function agent.init()
    -- DQN 네트워크 정의
    -- MLP(Multi Layer Perceptron)
    model = nn.Sequential()
    -- 입력 레이어
    model:add(nn.View(gridSize * gridSize))
    model:add(nn.Linear(gridSize * gridSize, 50))
    model:add(nn.ReLU())
    -- 히든 레이어
    model:add(nn.Linear(50, 50))
    model:add(nn.ReLU())
    -- 출력 레이어
    model:add(nn.Linear(50, numActions))
    model:add(nn.LogSoftMax())
    model:cuda()

    -- 손실 함수 정의
    criterion = nn.ClassNLLCriterion():cuda()
  end
  
    -- 학습된 모델 로드
  function agent.load(trainedModel)
    model = torch.load(trainedModel)
  end
  
  -- 액션 선택하기
  -- epsilon-greedy exploration
  function agent.chooseAction(state)
    local actionBy = ''    
    local action = nil
    if (util.randf(0, 1) <= epsilon) then
      while ( true ) do
        action = math.random(1, numActions)                    
        if ( util.isActionable(state, action) ) then
          actionBy = "exploration"
          break
        end                     
      end
     else       
       q = model:forward(state:view(-1))
       _, index = torch.sort(q, 1)
       for j = 1, 9 do
        action = index[-j]            
        if ( util.isActionable(state, action) ) then
          actionBy = "exploitation"
          break
        end            
       end      
    end
    
    --print(actionBy .. ": " .. action)
    
    return action
  end
  
  -- 에이전트 학습
  function agent.train(inputs, targets, sgdParams)
    local current_loss = 0.0
    local theta, gradTheta = model:getParameters()
    function feval(x_new)
      gradTheta:zero()
      local predictions = model:forward(inputs)
      local loss = criterion:forward(predictions, targets)
      local gradOutput = criterion:backward(predictions, targets)
      model:backward(inputs, gradOutput)
    
      return loss, gradTheta
    end

    local _, fs = optim.adam(feval, theta, sgdParams)
    current_loss = current_loss + fs[1]
    
    return current_loss
  end
  
  function agent.forward(inputs)
    return model:forward(inputs)
  end
  
  
  function agent.stone()
    return stone
  end
  
  function agent.getModel()
    return model
  end
  
  return agent
end
