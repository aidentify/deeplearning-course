require 'nngraph';
require 'pretty-nn';

torch.manualSeed(0)


-- input dimension
n = 3

-- output dimension
K = 1

-- first hidden layer dimensionality
d = 5

-- # of hidden layers
nHL = 2

-- length of longest sequences
-- unrolling steps
T = 4


xnode = - nn.Identity()
h1_t_1 = - nn.Identity()
h2_t_1 = - nn.Identity()

h1_t = {xnode, h1_t_1} - nn.JoinTable(1) - nn.Linear(n+d, d) - nn.Tanh()


h2_t = {h1_t, h2_t_1} - nn.JoinTable(1) - nn.Linear(2*d, d) - nn.Tanh()

ynode = h2_t - nn.Linear(d, K) - nn.Tanh()


rnn = nn.gModule({xnode, h1_t_1, h2_t_1}, {h1_t, h2_t, ynode})

graph.dot(rnn.fg, 'rnn', 'rnn')

x = torch.randn(n)
h0 = torch.zeros(d)
rnn:forward{x, h0, h0}

