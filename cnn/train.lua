require 'optim'
require 'gnuplot'

numIterPerEpoch = math.floor(trainingData:size(1)/opt.batchSize)

currentCount = 1

params, grad_params = model:getParameters()

currentClass = 0

--params:uniform(-0.005, 0.005)

function classification(output, targets)
  sizeTargets = targets:size(1)
  maxval, predClass = torch.max(output, 2)
  compareRes = torch.eq(targets:clone():double(), predClass:clone():double())
  numCorrect = torch.sum(compareRes)
  return numCorrect/sizeTargets
end

function validationLoss()
  val_inputs, val_targets, _ = 
  getBatch(testingData:size(1), testingData, testingLabels, 1)

  local output = model:forward(val_inputs:cuda())
  local loss = criterion:forward(output, val_targets:cuda():squeeze())
  local classificationVal = classification(output:float(), val_targets)

  return loss, classificationVal
end

function feval(x)
   -- get new parameters
   if x ~= params then
      params:copy(x)
   end

   -- reset gradients
   grad_params:zero()

   -- loss
   local loss = 0

   inputs, targets, currentCount = 
   getBatch(opt.batchSize, trainingData, trainingLabels, currentCount)

   -- evaluate function 
  local output = model:forward(inputs:cuda())

  local err = criterion:forward(output, targets:cuda():squeeze())

  loss = loss + err

  -- estimate df/dW
  local df_do = criterion:backward(output, targets:cuda():squeeze())
  model:backward(inputs:cuda(), df_do)

  currentClass = classification(output:float(), targets)

   -- return f and df/dX
   return loss,grad_params
end

maxValClass = 0

local optim_state = {learningRate = opt.lr, alpha = 0.95, epsilon = 1e-6}
for i=1,opt.epochs do
  avgLoss = 0
  for j=1, numIterPerEpoch do
    _, loss = optim.rmsprop(feval, params, optim_state)
    avgLoss = avgLoss + loss[1]
  end
  avgLoss = avgLoss/numIterPerEpoch
  valLoss, valClass = validationLoss()
  print(string.format("epoch %d, loss = %6.8f, gradnorm = %6.4e", i, avgLoss, grad_params:clone():norm()))
  print(string.format("validation loss = %6.8f, classification rate = %6.8f", valLoss, valClass)) 
 if valClass > maxValClass then
    torch.save(opt.modelfilename, model)
    maxValClass = valClass
  end

print(string.format("max classification rate = %6.8f", maxValClass))
  if not valclasses then
    valclasses = torch.Tensor(1)
    valclasses[1] = 1 - valClass
    valiter = torch.Tensor(1)
    valiter[1] = i
  else
    valclassesaddition = torch.Tensor(1)
    valclassesaddition[1] = 1 - valClass
    valiteraddition = torch.Tensor(1)
    valiteraddition[1] = i
    valclasses = torch.cat(valclasses:float(), valclassesaddition:float(), 1)
    valiter = torch.cat(valiter, valiteraddition, 1)
  end

  if not classes then
    classes = torch.Tensor(1)
    classes[1] = 1 - currentClass
    iter = torch.Tensor(1)
    iter[1] = i
  else
    classesaddition = torch.Tensor(1,1)
    classesaddition[1] = 1 - currentClass
    iteraddition = torch.Tensor(1)
    iteraddition[1] = i
    classes = torch.cat(classes:float(), classesaddition:float(),1)
    iter = torch.cat(iter, iteraddition, 1)
  end
if i%4 ==0 then
  gnuplot.pngfigure(opt.plotFilename)
  gnuplot.plot({'Training', iter, classes},{'Validation', valiter, valclasses})
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('classification error')
  gnuplot.plotflush()
end
end
