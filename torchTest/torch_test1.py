import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

device1=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train=torch.rand([50,1]).to(device1)
y_train=x_train*3+5
epoch=1000
class Model2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core=nn.Linear(1,1)
    def forward(self,x):
        out = self.core(x)
        return out

model=Model2().to(device1)
m_loss=nn.MSELoss()
optimizer1=optim.SGD(model.parameters(),lr=0.01)
x_axes=torch.arange(0,epoch)
y_axes=[]
for i1 in range(epoch):
    out=model(x_train)
    loss1=m_loss(out,y_train)
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()
    y_axes.append(loss1.detach())

# plt.plot(x_axes,y_axes)
# plt.show()
# plt.title('loss')

x_test=torch.tensor([[1],[2],[3]],dtype=torch.float32).to(device1)
model.eval()
y_predit=model(x_test)
plt.plot(x_test.cpu().detach().numpy() ,y_predit.cpu().detach().numpy())
plt.show()





        