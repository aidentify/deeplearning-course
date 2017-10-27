require 'nngraph';
require 'optim';

network = require 'network'

-- input dimension
n = 5

-- output dimension
K = 5

-- hidden layer dimensionality
d = 5

-- # of hidden layers
nHL = 1

-- length of longest sequences
-- unrolling steps
T = 6

--[[
-- Load RNN Model
]]--
model, prototype = network.getModel(n, d, nHL, K, T, 'RNN')

--graph.dot(net.fg, 'net', 'net')



--[[
-- 데이터 로드
--]]

x, y = nil

x_one_hot = torch.Tensor({{1, 0, 0, 0, 0},   --h
                          {0, 1, 0, 0, 0},   --i
                          {1, 0, 0, 0, 0},   --h
                          {0, 0, 1, 0, 0},   --e
                          {0, 0, 0, 1, 0},   --l
                          {0, 0, 0, 1, 0}   --l
                        })
                      
--x_one_hot = torch.Tensor({ {1, 0, 0, 0, 0} })
                      
y = torch.Tensor({2, 1, 3, 4, 4, 5}) -- ihello



-- Default intial state set to Zero
h0 = {}
h = {}
for l = 1, nHL do
   table.insert(h0, torch.zeros(d))
   table.insert(h, h0[#h0])
   if mode == 'FW' then -- Add the fast weights matrices A (A1, A2, ..., AnHL)
      table.insert(h0, torch.zeros(d, d))
      table.insert(h, h0[#h0])
   end
end


-- 손실 함수 정의
criterion = nn.ClassNLLCriterion()  

w, dE_dw = model:getParameters()


function table2Tensor(s)
   p = s[1]:view(1, 5)
   for t = 2, T do p =  p:cat(s[t]:view(1, 5), 1) end
   return p
end

function tensor2Table(inputTensor, padding)
   outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   for l = 1, padding do outputTable[l + inputTensor:size(1)] = h0[l] end
   return outputTable
end


function idx2char(idx)
  if idx == 1 then
    return 'h'
  end
  if idx == 2 then
    return 'i'
  end
  if idx == 3 then
    return 'e'
  end
  if idx == 4 then
    return 'l'
  end
  if idx == 5 then
    return 'o'
  end
  
end


--[[
-- 훈련하기
]]--


timer = torch.Timer()
trainError = 0

timer:reset()

for itr = 1, 2000 do
  
  feval = function()
    model:training()

    states = model:forward({x_one_hot, table.unpack(h)})
    -- States is the output returned from the selected models
    -- Contains predictions + tables of hidden layers
    -- {{y}, {h}}
    
    -- Store predictions
    prediction = table2Tensor(states)
    
    --print(prediction)
    _, max_idx = torch.max(prediction, 2)
    predict_seq = ''
    for idx = 1, 6 do
      predict_seq = predict_seq .. idx2char(max_idx[idx][1])
    end
    print(predict_seq)

    err = criterion:forward(prediction, y)
    --------------------------------------------------------------------------------
    -- Backward Pass
    --------------------------------------------------------------------------------
    dE_dh = criterion:backward(prediction, y)

    -- convert dE_dh into table and assign Zero for states
    m = mode == 'FW' and 2 or 1
    dE_dhTable = tensor2Table(dE_dh, m*nHL)

    model:zeroGradParameters()
    model:backward({xSeq, table.unpack(h)}, dE_dhTable)

    -- Store final output states
    for l = 1, nHL do h[l] = states[l + T] end
    if mode == 'FW' then for l = nHL+1, 2*nHL do h[l] = states[l + T] end end

    return err, dE_dw
 end

   w, err = optim.rmsprop(feval, w, optimState)
   trainError = trainError + err[1]   
   
   
  print(string.format("Iteration %8d, Training Error/seq_len = %4.4f, gradnorm = %4.4e",
                       itr, err[1] / T, dE_dw:norm()))

end
