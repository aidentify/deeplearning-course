require 'nngraph';
require 'optim';

--[[
-- Hyperparameters of network
]]--
-- input dimension
n = 4
-- output dimension
K = 4
-- hidden layer dimensionality
d = 3
-- # of hidden layers
nHL = 1
-- unrolling steps
T = 4

--[[
-- Load RNN Model
]]--
network = require 'network'
model, prototype = network.getModel(n, d, nHL, K, T, 'RNN')

--graph.dot(net.fg, 'net', 'net')



--[[
-- 데이터 로드
--]]
h = {1, 0, 0, 0}
e = {0, 1, 0, 0}
l = {0, 0, 1, 0}


x_one_hot = torch.Tensor({ h, e, l, l })
y = torch.Tensor({2, 3, 3, 4}) -- index of ello

print(x_one_hot)
print(y)


-- Default intial state set to Zero




h0 = {}
h = {}
for l = 1, nHL do
   table.insert(h0, torch.zeros(d))
   table.insert(h, h0[#h0])
end


-- 손실 함수 정의
criterion = nn.ClassNLLCriterion()  

w, dE_dw = model:getParameters()


function table2Tensor(s)
   p = s[1]:view(1, 4)
   for t = 2, T do p =  p:cat(s[t]:view(1, 4), 1) end
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
    return 'e'
  end
  if idx == 3 then
    return 'l'
  end
  if idx == 4 then
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
    for idx = 1, 4 do
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
