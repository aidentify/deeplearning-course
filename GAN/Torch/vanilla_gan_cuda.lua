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


--[[
-- x_real data 생성 및 n개 샘플링
-- 정규 분포 샘플링 N(4, 1.25)
-- 평균: 4.0 / 표준편차: 1.25
]]--
-- 평균/표준편차
data_mean = 4
data_stddev = 1.25

-- 정규분포 n개 샘플링
d_sampler = function(n)
  samples = torch.zeros(n):apply(
    function() i = torch.normal(data_mean, data_stddev); return i; end
  )
  return samples:cuda():sort()
end

x_real_t = d_sampler(1000)
gnuplot.hist(x_real_t)


--[[
-- Z noise 생성 및 m개 샘플링
]]--
-- Generator Input
gi_sampler = function(m)
  samples = torch.linspace(-8, 8, m) + torch.rand(m)*0.01
  return samples:view(m, 1):cuda()
end

z_noise_t = gi_sampler(1000)
gnuplot.hist(z_noise_t)


--[[
-- 구별망(Discriminator) 정의
]]--
-- 구별망 모델 파라미터
---- 입력 노드 개수
---- x real data에서 mini-batch개수 만큼 샘플링
d_input_size = 36             
d_hidden_size = 24
d_output_size = 1             -- 출력 노드 개수(1 or 0)
minibatch_size = d_input_size -- mini-batch size

-- 입력 레이어
netD = nn.Sequential()
netD:add(nn.Linear(d_input_size, d_hidden_size))
netD:add(nn.Tanh())
-- 히든 레이어
netD:add(nn.Linear(d_hidden_size, d_hidden_size))
netD:add(nn.Tanh())
-- 히든 레이어
netD:add(nn.Linear(d_hidden_size, d_hidden_size))
netD:add(nn.Tanh())
-- 출력 레이어
netD:add(nn.Linear(d_hidden_size, d_output_size))
netD:add(nn.Sigmoid())
netD:cuda()

-- 손실함수
criterion = nn.BCECriterion():cuda()


--[[
-- 생성망(Generator) 정의
]]--
-- 생성망 모델 파라미터
---- 입력 노드 개수
g_input_size = 1
g_hidden_size = 12
g_output_size = 1

-- 입력 레이어
netG = nn.Sequential()
netG:add(nn.Linear(g_input_size, g_hidden_size))
netG:add(nn.SoftPlus())
-- 출력 레이어
netG:add(nn.Linear(g_hidden_size, g_output_size))
netG:cuda()


--[[
-- 구별망 학습
-- logD(x) 최대화
-- (1-logD(G(z)) 최대화
]]--
-- 파라미터/그래디언트 초기화
parametersD, gradParametersD = netD:getParameters()
parametersG, gradParametersG = netG:getParameters()

-- CUDA용 텐서로 변환
parametersD = parametersD:cuda()
gradParametersD = gradParametersD:cuda()
parametersG = parametersG:cuda()
gradParametersG = gradParametersG:cuda()

-- 구별망 학습
fDx = function(x)
  gradParametersD:zero()
  -- logD(x) 최대화
  ---- x_real 데이터 샘플링
  d_real_data = d_sampler(d_input_size)
  ---- D(x) 목표값 예측
  d_real_decision = netD:forward(d_real_data)
  ---- 실제 레이블(1)
  d_real_label = torch.ones(1):cuda()
  d_real_error = criterion:forward(d_real_decision, d_real_label)
  df_do = criterion:backward(d_real_decision, d_real_label)
  netD:backward(d_real_data, df_do)
  
  -- (1-logD(G(z)) 최대화
  -- Z noise 샘플링
  d_gen_input = gi_sampler(minibatch_size)
  -- G(z) 데이터 생성
  d_fake_data = netG:forward(d_gen_input)
  d_fake_decision = netD:forward(d_fake_data:view(-1))
  ---- 실제 레이블(0)
  d_fake_label = torch.zeros(1):cuda()
  d_fake_error = criterion:forward(d_fake_decision, d_fake_label)
  df_do = criterion:backward(d_fake_decision, d_fake_label)
  netD:backward(d_real_data, df_do)
  
  d_error = d_real_error + d_fake_error
  return d_error, gradParametersD
end


--[[
-- 생성망 학습
-- (1-logD(G(z)) 최소화
-- logD(D(G(z)) 최대화
]]--
fGx = function(x)
  gradParametersG:zero():cuda()
  -- (1-logD(G(z)) 최소화
  -- logD(D(G(z)) 최대화
  ---- Z noise 샘플링
  gen_input = gi_sampler(minibatch_size)
  -- G(z) 데이터 생성
  g_fake_data = netG:forward(gen_input)
  dg_fake_decision = netD:forward(g_fake_data:view(-1))
  ---- 실제 레이블(1)
  dg_fake_label = torch.ones(1):cuda()
  g_error = criterion:forward(dg_fake_decision, dg_fake_label)
  df_do = criterion:backward(dg_fake_decision, dg_fake_label)
  df_dg = netD:updateGradInput(g_fake_data:view(-1), df_do)
  netG:backward(gen_input, df_dg[{ {1, minibatch_size} }]:view(minibatch_size, 1))
  
  return g_error, gradParametersG
end


--[[
-- 훈련 옵션
]]--
d_learning_rate = 2e-4
g_learning_rate = 2e-4
optim_betas = {0.9, 0.999}
num_epochs = 30000
print_interval = 200
save_interval = 5000
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
    learningRateDecay = learning_rate_decay,
    weightDecay = weight_decay,
    momentum = 0.9,
    dampening = 0,
    nesterov = true,
    beta1 = optim_betas[1],
    beta2 = optim_betas[2]
}


--[[
-- 구별망 pre-training
]]--
for d_index = 1, 3000 do
	optim.adam(fDx, parametersD, optimStateD)	
	print( ('Epoch #%d: d_real_error: %.3f d_fake_error: %.3f d_error: %.3f '):format(
            d_index,
            d_real_error, d_fake_error, d_error, g_error) )
end

--[[
-- 훈련하기
]]--
for epoch = 1, num_epochs do
  -- 구별망 학습
  for d_index = 1, d_steps do
    optim.adam(fDx, parametersD, optimStateD)
  end
  
  -- 생성망 학습
  for g_index = 1, g_steps do
    optim.adam(fGx, parametersG, optimStateG)
  end    
  
  -- 훈련 결과 출력
  if epoch % print_interval == 0 then
    print( ('Epoch #%d: d_real_error: %.3f d_fake_error: %.3f d_error: %.3f g_error: %.3f '
         .. 'd_mean: %.3f d_stddev: %.3f g_mean: %.3f g_stddev: %.3f'):format(
        epoch,
        d_real_error, d_fake_error, d_error, g_error,
        d_real_data:mean(), d_real_data:std(), g_fake_data:mean(), g_fake_data:std() ) )
    end
    
    -- 생성망 생성 결과 출력
    if epoch % save_interval == 0 then
      -- plotting generated distribution
      gen_x = gi_sampler(36000)
      g_fake_x = netG:forward(gen_x)
      gnuplot.pngfigure(epoch .. '.png')
      gnuplot.hist(g_fake_x)
      gnuplot.plotflush()       
      print( ('Epoch #%d: g_mean: %.3f g_stddev: %.3f'):format(
        epoch,
        g_fake_x:mean(), g_fake_x:std() ) )
    end
end
