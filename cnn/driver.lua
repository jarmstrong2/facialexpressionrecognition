require 'getBatch'
require 'cunn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Facial Expression Recognition')
cmd:text()
cmd:text('Options:')
cmd:option('-batchSize', '64', 'batch size for training')
cmd:option('-lr', '1e-3', 'learning rate')
cmd:option('-epochs', '200', 'epochs of training')
cmd:option('-datafile', 'crossval_1.mat', 'data for training and testing')
cmd:option('-modelfilename', 'cnnFER.t7', 'model file name')
cmd:option('-plotFilename', 'FERplot.png', 'plot file name')
cmd:option('-encoderFile', 'something.t7', 'encoder file name')
cmd:text()
opt = cmd:parse(arg or {})

dofile('data.lua')
dofile('model.lua')
dofile('train.lua')
