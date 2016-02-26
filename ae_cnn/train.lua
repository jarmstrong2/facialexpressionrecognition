require 'optim'
require 'gnuplot'

numIterPerEpoch = math.floor(noiseData:size(1)/2)

currentCount = 1

params, grad_params = ae:getParameters()

params:uniform(-0.05, 0.05)

function feval(x)
   -- get new parameters
   if x ~= params then
      params:copy(x)
   end

   -- reset gradients
   grad_params:zero()

   -- loss
   local loss = 0

   inputs, currentCount = 
   getBatch(2, noiseData, currentCount)

   -- evaluate function 
  local output = ae:forward(inputs:cuda())

  local err = criterion:forward(output, inputs:cuda())

  print(loss)
  loss = loss + err

  -- estimate df/dW
  local df_do = criterion:backward(output, inputs:cuda())
  ae:backward(inputs:cuda(), df_do)

   -- return f and df/dX
   return loss,grad_params
end

minLoss = (1/0)

local optim_state = {learningRate = 1e-4, alpha = 0.95, epsilon = 1e-6}
for i=1,200 do
  avgLoss = 0
  for j=1, numIterPerEpoch do
    _, loss = optim.rmsprop(feval, params, optim_state)
    avgLoss = avgLoss + loss[1]
    print(string.format("current loss = %6.8f", loss[1]))
  end
  avgLoss = avgLoss/numIterPerEpoch
  print(string.format("epoch %d, loss = %6.8f, gradnorm = %6.4e", i, avgLoss, grad_params:clone():norm()))
  if avgLoss < minLoss then
    torch.save('encoder.t7', encoder)
    minLoss = avgLoss
  end

  if not losses then
    losses = torch.Tensor(1)
    losses[1] = avgLoss
    iter = torch.Tensor(1)
    iter[1] = i
  else
    lossesaddition = torch.Tensor(1,1)
    lossesaddition[1] = avgLoss
    iteraddition = torch.Tensor(1)
    iteraddition[1] = i
    losses = torch.cat(losses:float(), lossesaddition:float(),1)
    iter = torch.cat(iter, iteraddition, 1)
  end

  gnuplot.pngfigure('AEplot.png')
  gnuplot.plot({'Training', iter, losses})
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  gnuplot.plotflush()

end