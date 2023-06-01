# This is a sample Python script.

import  torch
from mAttention import Multi_attention1
from mECell import Eecoder_Cell
from mDCell import Decoder_cell
from mNet import Net_Transformer
from torch.optim import SGD
from mData_Loader import data_loader_train1
from mConfig import Config1
import torch.nn.functional as F
import  numpy as np
import matplotlib.pyplot as plt


config1=Config1()


# Test Filed
if __name__ == '__main__':

    device1 = torch.device('cuda')
    model1 = Net_Transformer(4,196)# size depend on data shape
    model1.to(device1)
    optimizer1=SGD(model1.parameters(),lr=config1.LEARNING_RATE,momentum=config1.MOMENTUM)
    loss_list=[]
    for _ in range(config1.EPOCH):
        for i,all_data in enumerate(data_loader_train1):
            data1,label1=all_data
            data1 = data1.squeeze(1).squeeze(0)
            # data1 = data1[:,10:18,10:18]

            data_a,data_b=data1[0:14,0:14].reshape(1,-1),data1[0:14,14:28].reshape(1,-1)
            data_c,data_d=data1[14:28,0:14].reshape(1,-1),data1[14:28,14:28].reshape(1,-1)
            data1=torch.cat([data_a,data_b,data_c,data_d],dim=0)
            data1 = data1.to(device1)
            label1=label1.to(device1)
            label1=label1.repeat([4])
            output1=model1(data1)
            loss1=F.cross_entropy(output1,label1)
            #loss1=F.nll_loss(output1,label1)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            if(i%5==0):
                loss_list.append(loss1.item())
                print(loss1.item())
            if(i>800):
                break
            # if(loss1.item()<1):
            #     torch.save(model1.state_dict(), 'model1_state.pth')
    x_axse=np.arange(len(loss_list))
    plt.plot(x_axse,loss_list,label='simple transformer without decoder')
    plt.xlabel('num iter')
    plt.ylabel('loss')
    plt.legend()
    plt.show()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
