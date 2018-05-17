local Ravel, parent = torch.class('nn.Ravel', 'nn.Module')
function Ravel:__init(seq_length)
  parent.__init(self)
  self.seq_length = seq_length
end
function Ravel:updateOutput(input)
  assert(input:nDimension() == 1)
  self.output:resize(input:size(1)/self.seq_length, self.seq_length)
  for k=1,self.output:size(1) do
     self.output[k] = input[{{(k-1)*self.seq_length+1, k*self.seq_length}}]
  end
  return self.output
end
function Ravel:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  for k=1,gradOutput:size(1) do
     self.gradInput[{{(k-1)*self.seq_length+1, k*self.seq_length}}] = gradOutput[k]
  end
  return self.gradInput
end
