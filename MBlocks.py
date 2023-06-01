from torch import nn
import torch.nn.functional as F
import torch

class MBLOCK1(nn.Module):
    def __init__(self,channel1,*,channel2=256) -> None:
        super().__init__()
        self.channel1=channel1
        self.channel2 = channel2
        self.conv1=nn.Conv2d(self.channel1,self.channel2,3,1,1)
        self.conv2=nn.Conv2d(self.channel2,self.channel1,3,1,1)
        self.bn1=nn.BatchNorm2d(self.channel2)
        self.bn2=nn.BatchNorm2d(self.channel1)
    def forward(self,x):
        fx1=self.conv1(x)
        fx1=self.bn1(fx1)
        fx1=F.relu(fx1)
        
        fx1=self.conv2(fx1)
        fx1=self.bn2(fx1)

        return F.relu(fx1+x)

if __name__=='__main__':
    x1=torch.rand((1,1,28,28))
    model1=MBLOCK1(1)
    output1=model1(x1)
    print(output1.shape)

