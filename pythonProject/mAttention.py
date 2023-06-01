import torch
from torch import  nn
import torch.nn.functional as F

from mConfig import Config1
config1 = Config1()

class Attention1(nn.Module):
    def __init__(self,height1,weight1,*,weight2=config1.INNER_FEATURE_A,mask_flag=False,combination_flag=False):
        super().__init__()
        self.height = height1
        #self.mask_flag=mask_flag
        self.combi_flag=combination_flag
        self.Wq1 = nn.Parameter(torch.randn(weight1, weight2))
        self.Wk1 = nn.Parameter(torch.randn(weight1, weight2))
        self.Wv1 = nn.Parameter(torch.randn(weight1, weight2))
        self.mask1 = torch.ones((height1,height1)) if mask_flag is False else  self.fgenerate_mask(height1)

    def forward(self,input):
        if self.combi_flag == False:
            Q1=torch.matmul(input,self.Wq1)
            K1=torch.matmul(input,self.Wq1)
            V1=torch.matmul(input,self.Wq1)
        else:
            Q1=torch.matmul(input[0],self.Wq1)
            K1=torch.matmul(input[1],self.Wq1)
            V1=torch.matmul(input[2],self.Wq1)


        attention_matrix=torch.matmul(Q1,torch.transpose(K1,1,0))
        attention_matrix = attention_matrix*self.mask1
        attention_matrix=F.softmax(attention_matrix,dim=1)

        Z1 = torch.matmul(attention_matrix,V1) #size=(self.height,256)
        return Z1

    def to(self,device1):
        super().to(device1)
        self.Wq1=self.Wq1.to(device1)
        self.Wk1=self.Wk1.to(device1)
        self.Wv1=self.Wv1.to(device1)
        self.mask1=self.mask1.to(device1)

    def fgenerate_mask(self,height1):
        mask=torch.ones((height1,height1))
        return torch.tril(mask,diagonal=0)


class Multi_attention1(nn.Module):
    def __init__(self,height1,weight1,*,num_Head=4,mflag=False,c_flag=False):
        super().__init__()
        self.attenion1 = Attention1(height1,weight1,mask_flag=mflag,combination_flag=c_flag)
        self.attenion2 = Attention1(height1, weight1,mask_flag=mflag,combination_flag=c_flag)
        self.attenion3 = Attention1(height1, weight1,mask_flag=mflag,combination_flag=c_flag)
        self.attenion4 = Attention1(height1, weight1,mask_flag=mflag,combination_flag=c_flag)

        self.linear1=torch.nn.Linear(config1.INNER_FEATURE_MA,weight1)

    def forward(self,input):
        input1 = self.attenion1(input)
        input2 = self.attenion2(input)
        input3 = self.attenion3(input)
        input4 = self.attenion4(input)
        x = torch.cat([input1,input2,input3,input4],1)
        x = self.linear1(x)
        return  x
    def to(self,device1):
        super().to(device1)
        self.linear1.to(device1)
        self.attenion1.to(device1)
        self.attenion2.to(device1)
        self.attenion3.to(device1)
        self.attenion4.to(device1)












