local Print, parent = torch.class('nn.Print', 'nn.Module')
function Print:__init()
  parent.__init(self)
end
function Print:updateOutput(input)
   print("Print ... ")
   local scores = input:reshape(10,16)
   print(scores)
   self.output = input
   return self.output
end
function Print:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end
