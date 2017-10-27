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


--[[
-- utility functions
]]--
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

function table2Tensor(s)
   p = s[1]:view(1, 4)
   for t = 2, T do p =  p:cat(s[t]:view(1, 4), 1) end
   return p
end


--[[
-- 1st Feed fowarding
]]--
-- Default intial state set to Zero
h0 = torch.zeros(d)
h = h0
print(h0)

model:training()

-- States is the output returned from the selected models
-- Contains predictions + tables of hidden layers
-- {{y}, {h}}
states = model:forward({x_one_hot, h})
print(states)

-- Store predictions
prediction = table2Tensor(states)
print(prediction)

--print(prediction)
_, max_idx = torch.max(prediction, 2)
predict_seq = ''
for idx = 1, 4 do
  predict_seq = predict_seq .. idx2char(max_idx[idx][1])
end
print(predict_seq)

-- Store final hidden states for next interation
h = states[T + 1]
print(h)
