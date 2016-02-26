require 'torch'
require 'image'
require 'nn' 

matio = require 'matio'

noiseData = (matio.load('../ae/unlabeled_images.mat', 'unlabeled_images')):double()

noiseData = nn.Transpose({1,3}):forward(noiseData)
noiseData:resize(noiseData:size(1), 1, noiseData:size(2), noiseData:size(3))
noiseData:float()

-- normalize data
mean = noiseData:mean()
std = noiseData:std()

noiseData:add(-mean)
noiseData:div(std)

noiseData = noiseData[{{1,1000},{},{},{}}]
collectgarbage()
