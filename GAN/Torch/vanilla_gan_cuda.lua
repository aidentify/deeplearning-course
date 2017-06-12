--[[
-- Vanilla GAN implementatoin 
-- Generating normal distribution N(4.0, 1.25)
@author socurites@gmail.com
]]--

require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'
require 'cutorch'
require 'cunn'
require 'cudnn'

-- Data params
data_mean = 4
data_stddev = 1.25

-- Model params
d_input_size = 300   -- Minibatch size - cardinality of distributions
d_hidden_size = 25   -- Discriminator complexity
d_output_size = 1    -- Single dimension for 'real' vs. 'fake'
minibatch_size = d_input_size

g_input_size = 1     -- Random noise dimension coming into generator, per output vector
g_hidden_size = 10   -- Generator complexity
g_output_size = 1    -- size of generated output vector

-- Learning hyper-parameters
d_learning_rate = 2e-4  -- 2e-4
g_learning_rate = 2e-4
optim_betas = {0.9, 0.999}
num_epochs = 60000000
print_interval = 200
save_interval = 5000
-- 'k' steps in the original GAN paper.
-- Can put the discriminator on higher training freq than generator
d_steps = 1
g_steps = 1


learning_rate_decay = 1e-9
weight_decay = 0.0

optimStateD = {
    learningRate = d_learning_rate,
    learningRateDecay = learning_rate_decay,
    weightDecay = weight_decay,
    momentum = 0.9,
    dampening = 0,
    nesterov = true,
    beta1 = optim_betas[1],
    beta2 = optim_betas[2]
}
optimStateG = {
    learningRate = g_learning_rate,
    momentum = 0.9,
    dampening = 0,
    nesterov = true,
    beta1 = optim_betas[1],
    beta2 = optim_betas[2]
}



-- Gaussian
d_sampler = function(n)
    samples = torch.zeros(n):apply(function() i = torch.normal(data_mean, data_stddev); return i; end)
    return samples:cuda():sort()
end

-- Generator Input
gi_sampler = function(m, n)
    --samples = torch.rand(m, n):view(-1):sort()
    --return samples:view(m, n)
    samples = torch.linspace(-8, 8, m) + torch.rand(m)*0.01
    return samples:view(m, n):cuda()
end

decorate_with_diffs = function(data, exponent)
    --mean = torch.mean(data, 1)[1]
    --diffs = torch.pow(data - mean, exponent)
    --return torch.cat(data, diffs, 1)
    return data
end

preprocess = function(data)
    return decorate_with_diffs(data, 2.0)
end



-- Generator model
netG = nn.Sequential()
netG:add(nn.Linear(g_input_size, g_hidden_size))
netG:add(nn.SoftPlus())
netG:add(nn.Linear(g_hidden_size, g_output_size))
netG:cuda()

-- Discriminator model
netD = nn.Sequential()
netD:add(nn.Linear(d_input_size, d_hidden_size))
netD:add(nn.Tanh())
netD:add(nn.Linear(d_hidden_size, d_hidden_size))
netD:add(nn.Tanh())
netD:add(nn.Linear(d_hidden_size, d_output_size))
netD:add(nn.Sigmoid())
netD:cuda()

-- Binary cross entropy
criterion = nn.BCECriterion():cuda()

parametersD, gradParametersD = netD:getParameters()
parametersG, gradParametersG = netG:getParameters()

parametersD = parametersD:cuda()
gradParametersD = gradParametersD:cuda()
parametersG = parametersG:cuda()
gradParametersG = gradParametersG:cuda()

-- create closure to evaluate f(X) and df/dX of discriminator
fDx = function(x)
    gradParametersD:zero()
    -- train with real
    d_real_data = d_sampler(d_input_size)
    d_real_decision = netD:forward(preprocess(d_real_data))
    d_real_label = torch.ones(1):cuda()
    d_real_error = criterion:forward(d_real_decision, d_real_label)
    df_do = criterion:backward(d_real_decision, d_real_label)
    netD:backward(preprocess(d_real_data), df_do)
   -- train with fake
    d_gen_input = gi_sampler(minibatch_size, g_input_size)
    d_fake_data = netG:forward(d_gen_input)
    d_fake_decision = netD:forward(preprocess(d_fake_data:view(-1)))
    d_fake_label = torch.zeros(1):cuda()
    d_fake_error = criterion:forward(d_fake_decision, d_fake_label)
    df_do = criterion:backward(d_fake_decision, d_fake_label)
    netD:backward(preprocess(d_real_data), df_do)
    d_error = d_real_error + d_fake_error
    return d_error, gradParametersD
end


fGx = function(x)
    gradParametersG:zero():cuda()
    gen_input = gi_sampler(minibatch_size, g_input_size)
    g_fake_data = netG:forward(gen_input)
    dg_fake_decision = netD:forward(preprocess(g_fake_data:view(-1)))
    dg_fake_label = torch.ones(1):cuda()
    g_error = criterion:forward(dg_fake_decision, dg_fake_label)
    df_do = criterion:backward(dg_fake_decision, dg_fake_label)
    df_dg = netD:updateGradInput(preprocess(g_fake_data:view(-1)), df_do)
    netG:backward(gen_input, df_dg[{ {1, minibatch_size} }]:view(minibatch_size, 1))
   return g_error, gradParametersG
end


for d_index = 1, 3000 do
	optim.adam(fDx, parametersD, optimStateD)	
	print( ('Epoch #%d: d_real_error: %.3f d_fake_error: %.3f d_error: %.3f '):format(
            d_index,
            d_real_error, d_fake_error, d_error, g_error) )
end

-- train
for epoch = 1, num_epochs do
    for d_index = 1, 3 do
        optim.adam(fDx, parametersD, optimStateD)
    end
    for g_index = 1, g_steps do
        optim.adam(fGx, parametersG, optimStateG)
    end    
    if epoch % print_interval == 0 then
        print( ('Epoch #%d: d_real_error: %.3f d_fake_error: %.3f d_error: %.3f g_error: %.3f '
             .. 'd_mean: %.3f d_stddev: %.3f g_mean: %.3f g_stddev: %.3f'):format(
            epoch,
            d_real_error, d_fake_error, d_error, g_error,
            d_real_data:mean(), d_real_data:std(), g_fake_data:mean(), g_fake_data:std() ) )
    end
    if epoch % save_interval == 0 then
        -- plotting generated distribution
        gen_x = gi_sampler(60000, g_input_size)
        g_fake_x = netG:forward(gen_x)
        gnuplot.pngfigure(epoch .. '.png')
        gnuplot.hist(g_fake_x)
        gnuplot.plotflush()
        -- plotting decision boundary
        dx = gen_x:view(-1)
        dy = g_fake_x:view(-1)
        x_val = torch.zeros(200):cuda()
        y_val = torch.zeros(200):cuda()
        k = 1
        for i = 1, 60000, 300 do            
            j = i + 300 - 1
            dg_fd = netD:forward(preprocess(dy[ { {i,j} } ]))
            x_val[k] = dx[ { {i,j} } ]:mean()
            y_val[k] = dg_fd
            k = k+1
        end
        gnuplot.pngfigure(epoch .. '_db.png')
        gnuplot.hist(g_fake_x)
        gnuplot.plot('d', x_val, y_val, '~')
        gnuplot.plotflush()
        print( ('Epoch #%d: g_mean: %.3f g_stddev: %.3f'):format(
            epoch,
            g_fake_x:mean(), g_fake_x:std() ) )
    end
end
