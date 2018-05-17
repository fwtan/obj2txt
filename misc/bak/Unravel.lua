local Unravel, parent = torch.class('nn.Unravel', 'nn.Module')
function Unravel:__init()
  parent.__init(self)
end
function Unravel:updateOutput(input)
  assert(input:nDimension() == 3)
  self.output:resize(input:size(1)*input:size(2), input:size(3))
  for k=1,input:size(1) do
     self.output[{ {(k-1)*input:size(2)+1, k*input:size(2)}, {} }] = input[{ k, {}, {} }]
  end
  return self.output
end
function Unravel:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  for k=1,input:size(1) do
     self.gradInput[{ k, {}, {} }] = gradOutput[{ {(k-1)*input:size(2)+1, k*input:size(2)}, {} }]
  end
  return self.gradInput
end
