import torch
from torch import  nn
import torch.nn.functional as F
from mAttention import Multi_attention1
from mConfig import Config1

config1=Config1()

class Eecoder_Cell(nn.Module):
    def __init__(self,heigh1,weight1,*,weight2=config1.INNER_FEATURE_E):
        super().__init__()
        self.weight=weight1
        self.mattention=Multi_attention1(heigh1,weight1)
        self.layer_feed1=nn.Sequential(
            nn.Linear(weight1,weight2),
            nn.ReLU(),
            nn.Linear(weight2,weight1)
        )
    def forward(self,input1):
        x1 = self.mattention(input1)
        x1 = F.layer_norm((x1+input1),(self.weight,))
        x2=self.layer_feed1(x1)
        x2=F.layer_norm((x2+x1),(self.weight,))
        return x2

    def to(self,device1):
        super().to(device1)
        self.mattention.to(device1)
        self.layer_feed1.to(device1)

