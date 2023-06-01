import torch

from modulex import  Net1

model1 = Net1()
model1.load_state_dict(torch.load('model1_state.pth'))

example_forward_input=torch.rand((1,28,28))
# model1(input1)
model1_jit=torch.jit.trace(model1,example_forward_input)
torch.jit.save(model1_jit,'model1_jit.pth')
