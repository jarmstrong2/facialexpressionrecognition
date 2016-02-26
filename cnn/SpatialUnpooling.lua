local SpatialUnpooling, parent = torch.class('nn.SpatialUnpooling', 'nn.Module')

function SpatialUnpooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self)
   self.dW = dW or kW
   self.dH = dH or kH
   self.padW = padW or 0
   self.padH = padH or 0
   self.indices = torch.LongTensor()
   self._indexTensor = torch.LongTensor()
end

function SpatialUnpooling:updateOutput(input)
  local n, d, h, w, oh, ow 
  if input:nDimension() == 4 then -- batch
    n, d, h, w = input:size(1), input:size(2), input:size(3), input:size(4)
    oh, ow = h * self.dH + 2 * self.padH, w * self.dW + 2 * self.padW
    self.output:resize(n, d, oh, ow)
  else
    n, d, h, w = 1, input:size(1), input:size(2), input:size(3)
    oh, ow = h * self.dH + 2 * self.padH, w * self.dW + 2 * self.padW
    self.output:resize(d, oh, ow)
  end

  local in_cols, out_cols, rows = h * w, oh * ow, n * d
  self.output:zero()
  self.output:view(rows, out_cols):scatter(
    2, 
    self.indices:view(rows, in_cols):typeAs(self._indexTensor), 
    input:view(rows, in_cols)
  )
  return self.output
end

function SpatialUnpooling:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  local n, d, h, w, oh, ow 
  if input:nDimension() == 4 then -- batch
    n, d, h, w, oh, ow = input:size(1), input:size(2), input:size(3), input:size(4), gradOutput:size(3), gradOutput:size(4)
  else
    n, d, h, w, oh, ow = 1, input:size(1), input:size(2), input:size(3), gradOutput:size(2), gradOutput:size(3)
  end

  local in_cols, out_cols, rows = h * w, oh * ow, n * d
  self.gradInput:view(rows, in_cols):gather(
    gradOutput:view(rows, out_cols),
    2, 
    self.indices:view(rows, in_cols):typeAs(self._indexTensor)
  )
  return self.gradInput
end

function SpatialUnpooling:type(type, tensorCache)
  parent.type(self, type, tensorCache)
  if type == 'torch.CudaTensor' then
    self._indexTensor:type(type)
  else
    self._indexTensor = torch.LongTensor()
  end
end
   
function SpatialUnpooling:__tostring__()
   return string.format('%s(%d,%d)', torch.type(self), self.kW, self.kH)
end