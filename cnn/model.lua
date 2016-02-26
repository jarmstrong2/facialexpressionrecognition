require 'torch'  
require 'image'   
require 'nn'
require 'cunn'

encoder = torch.load(opt.encoderFile)

model = nn.Sequential()

model:add(encoder)
model:add(nn.LogSoftMax())
model = model:cuda()

criterion = nn.ClassNLLCriterion():cuda()
