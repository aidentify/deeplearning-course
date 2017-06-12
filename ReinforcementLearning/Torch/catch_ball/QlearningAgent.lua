--[[
Reinforcement Learning / Deep Q-Learning Example
Qlearning agent

@author socurites@gmail.com
]]--
function QlearningAgent(gridSize, numAction, epsilon, epsilonDecay, epsilonMinimumValue)
  local agent = {}
  local model
  local criterion
  
  function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
  end
  
  -- Qlearning agent 초기화
  function agent.init()
    -- DQN 네트워크 정의
    -- MLP(Multi Layer Perceptron)
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
  end
  
  -- 학습된 모델 로드
  function agent.load(trainedModel)
    model = torch.load(trainedModel)
  end
  
  -- 액션 선택하기
  -- epsilon-greedy exploration
  function agent.chooseAction(state)
    local actionBy = ''

    if (randf(0, 1) <= epsilon) then
      action = math.random(1, numAction)
      actionBy = "exploration"
    else
      q = model:forward(state)
      _, index = torch.max(q, 1)
      action = index[1]
      actionBy = "exploitation"
    end

    --print(actionBy .. ": " .. action)

    -- Decay the epsilon
    if (epsilon > epsilonMinimumValue) then
      epsilon = epsilon * epsilonDecay
    end
    
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

    local _, fs = optim.sgd(feval, theta, sgdParams)
    current_loss = current_loss + fs[1]
    
    return current_loss
  end
  
  function agent.forward(inputs)
    return model:forward(inputs)
  end
  
  function agent.getModel()
    return model
  end
  
  return agent
end