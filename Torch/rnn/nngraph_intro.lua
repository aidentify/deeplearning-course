require 'nngraph';
require 'pretty-nn';

torch.manualSeed(0)

--[[
-- nn 패키지로 네트워크 정의하기
--]]
net = nn.Sequential();
net:add(nn.Linear(20, 10));
net:add(nn.Tanh());
net:add(nn.Linear(10, 10));
net:add(nn.Tanh());
net:add(nn.Linear(10, 1));

print(net)


--[[
-- 그래프 노드(nngraph.node) 정의하기
--]]
h1 = net.modules[1]()

print(h1)

print({h1})

net:get(1).bias:view(1, -1)

h1.data.module.bias:view(1, -1)


-- 출력 노드
h2 = net.modules[5](net.modules[4](net.modules[3](net.modules[2](h1))))


gNet = nn.gModule({h1}, {h2})

print(gNet)

-- 네트워크 그래프 저장
graph.dot(gNet.fg, 'mlp', 'mlp')


--[[
-- 그래프 노드(nngraph.node) 정의하기
-- - notation
--]]
g1 = - nn.Linear(20, 10)
g2 = g1 - nn.Tanh() - nn.Linear(10, 10) -  nn.Tanh() - nn.Linear(10, 1)

mlp = nn.gModule({g1}, {g2})

graph.dot(mlp.fg, 'mlp2', 'mlp2')



--[[
-- 그래프 노드(nngraph.node) 정의하기
-- nn.JoinTable
--]]
input = - nn.Identity()

L1 = input - nn.Linear(10, 20) - nn.Tanh()

L2 = {input, L1} - nn.JoinTable(1) - nn.Linear(30, 60) - nn.Tanh()

L3 = {L1, L2} - nn.JoinTable(1) - nn.Linear(80, 1) - nn.Tanh()

g = nn.gModule({input}, {L3})


graph.dot(g.fg, 'fancy', 'fancy')