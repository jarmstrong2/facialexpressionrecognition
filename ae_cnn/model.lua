require 'nn'
require 'torch'
require 'SpatialUnpooling'
require 'cunn'

-- 7 class facial expression problem
noutputs = 7

-- input specifications
nfeats = 1
width = 32
height = 32

ninputs = nfeats*width*height
nstates = {96,128,128,1024}
filtsize_1 = 5
filtsize_all = 3
poolsize = 2

poollayer1 = nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)
unpoollayer1 = nn.SpatialUnpooling(poolsize,poolsize,poolsize,poolsize)
poollayer1:cuda()
unpoollayer1:cuda()
unpoollayer1.indices = poollayer1.indices

poollayer2 = nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)
unpoollayer2 = nn.SpatialUnpooling(poolsize,poolsize,poolsize,poolsize)
poollayer2:cuda()
unpoollayer2:cuda()
unpoollayer2.indices = poollayer2.indices

encoder = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling
encoder:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize_1, filtsize_1))
encoder:add(nn.PReLU())
encoder:add(poollayer1)

-- stage 2 : filter bank -> squashing
encoder:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize_all, filtsize_all))
encoder:add(nn.PReLU())

-- stage 3 : filter bank -> squashing -> L2 pooling
encoder:add(nn.SpatialConvolution(nstates[2], nstates[2], filtsize_all, filtsize_all))
encoder:add(nn.PReLU())

-- stage 4 : filter bank -> squashing -> L2 pooling
encoder:add(nn.SpatialConvolution(nstates[2], nstates[3], filtsize_all, filtsize_all))
encoder:add(nn.PReLU())

encoder:add(poollayer2)

-- stage 5 : standard 2-layer neural network
encoder:add(nn.View(nstates[3]*4*4))
encoder:add(nn.Dropout(0.5))
encoder:add(nn.Linear(nstates[3]*4*4, nstates[4]))
encoder:add(nn.Dropout(0.5))
encoder:add(nn.Linear(nstates[4], nstates[4]))
encoder:add(nn.PReLU())
encoder:add(nn.Dropout(0.5))
encoder:add(nn.Linear(nstates[4], noutputs))

decoder = nn.Sequential()

-- stage 6 : standard 2-layer neural network
decoder:add(nn.Dropout(0.5))
decoder:add(nn.Linear(noutputs, nstates[4]))
decoder:add(nn.Dropout(0.5))
decoder:add(nn.PReLU())
decoder:add(nn.Linear(nstates[4], nstates[4]))
decoder:add(nn.Dropout(0.5))
decoder:add(nn.Linear(nstates[4], nstates[3]*4*4))
decoder:add(nn.Dropout(0.5))
decoder:add(nn.View(nstates[3], 4, 4))

-- stage 7 : filter unpooling -> squashing -> filter bank
decoder:add(unpoollayer2)
decoder:add(nn.PReLU())
decoder:add(nn.SpatialFullConvolution(nstates[3], nstates[2], filtsize_all, filtsize_all))

-- stage 8 : filter unpooling -> squashing -> filter bank
decoder:add(nn.PReLU())
decoder:add(nn.SpatialFullConvolution(nstates[2], nstates[2], filtsize_all, filtsize_all))

-- stage 9 : squashing -> filter bank
decoder:add(nn.PReLU())
decoder:add(nn.SpatialFullConvolution(nstates[2], nstates[1], filtsize_all, filtsize_all))

-- stage 10 : filter unpooling -> squashing -> filter bank
decoder:add(unpoollayer1)
decoder:add(nn.PReLU())
decoder:add(nn.SpatialFullConvolution(nstates[1], nfeats, filtsize_1, filtsize_1))

ae = nn.Sequential()
ae:add(nn.Dropout(0.9))
ae:add(encoder)
ae:add(decoder)
ae = ae:cuda()

criterion = nn.MSECriterion():cuda()