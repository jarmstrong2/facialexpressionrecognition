require 'torch'
require 'image'
require 'nn' 

matio = require 'matio'

data = matio.load(opt.datafile)
trainingData = data.trainingPts
trainingLabels = data.trainingLabels
testingData = data.testingPts
testingLabels = data.testingLabels

-- transpose so that the batch size is the first index
trainingData = nn.Transpose({1,3}):forward(trainingData)
trainingData:resize(trainingData:size(1), 1, trainingData:size(2), trainingData:size(3))
trainingData:float()

testingData = nn.Transpose({1,3}):forward(testingData)
testingData:resize(testingData:size(1), 1, testingData:size(2), testingData:size(3))
testingData:float()

-- normalize data
mean = trainingData:mean()
std = trainingData:std()

trainingData:add(-mean)
trainingData:div(std)

testingData:add(-mean)
testingData:div(std)

