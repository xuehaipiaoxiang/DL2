import torch
import torch.nn.functional as F
from mAttention import Multi_attention1
from torch import nn
from mConfig import Config1

config1 = Config1()

class Decoder_cell(nn.Module):
    def __init__(self,heigh1,weight1,*,weight2=config1.INNER_FEATURE_D):
        super().__init__()
        self.height,self.weight=heigh1,weight1

        self.mattention1=Multi_attention1(heigh1,weight1,mflag=True)
        self.mattention2=Multi_attention1(heigh1,weight1,c_flag=True)

        self.layer_feed1=nn.Sequential(
            nn.Linear(weight1,weight2),
            nn.ReLU(),
            nn.Linear(weight2,weight1)
        )


    def forward(self,inputs2):
        '''
        inputs2::=[Xdata,EncoderData]]
        '''
        input1,e_input1=inputs2
        x1=self.mattention1(input1)
        x1 = F.layer_norm((x1 + input1), (self.weight,))

        x2=self.mattention2((e_input1,e_input1,x1))
        x3 = F.layer_norm((x1 + x2), (self.weight,))

        x4=self.layer_feed1(x3)
        x4= F.layer_norm((x3 + x4), (self.weight,))
        return  x4

    def to(self,device1):
        super().to(device1)
        self.mattention1.to(device1)
        self.mattention2.to(device1)
        self.layer_feed1.to(device1)


