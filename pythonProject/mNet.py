import  torch

from mECell import Eecoder_Cell
from mDCell import Decoder_cell
from torch import nn
from mConfig import Config1
import torch.nn.functional as F

config1=Config1()


class Net_Transformer(nn.Module):
    def __init__(self,height,weight):
        super().__init__()
        self.encoder1 = Eecoder_Cell(height,weight)
        self.encoder2 = Eecoder_Cell(height, weight)
        self.decoder1 = Decoder_cell(height,weight)
        self.decoder2 = Decoder_cell(height, weight)
        self.linear1 = nn.Linear(weight,config1.FINAL_CLASSES)
    def forward(self,x):
        x1=self.encoder1(x)
        x1=self.encoder2(x1)

       # x2=self.decoder1((x,x1))
       # x2=self.decoder2((x2,x1))


        x3=self.linear1(x1) #x5.shape(height1=num_Word,FINAL_CLASSES=num_class)

       # x3 = F.softmax(x3,dim=1)
        #x3 = F.log_softmax(x3,dim=1)
        return x3



    def to(self,device1):
        super().to(device1)
        self.encoder1.to(device1)
        self.encoder2.to(device1)
        self.decoder1.to(device1)
        self.decoder2.to(device1)
        self.linear1.to(device1)


