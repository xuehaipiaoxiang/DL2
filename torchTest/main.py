import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.optim import SGD
from modulex import Net1
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

mbatch_size=128
mlearning_rate=0.01
mmomentum=0.5
miternel=5
mepoch=5

device1=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_train=MNIST('./data',train=True,transform=
               Compose([ToTensor(),Normalize(0.1307,0.3081)]),
                download = True
)

dataset_test=MNIST('./data',train=True,transform=
               Compose([ToTensor(),Normalize(0.1,0,3)]),
)

data_loader_train=DataLoader(dataset=dataset_train,batch_size=mbatch_size,shuffle=True)
data_loader_test=DataLoader(dataset=dataset_test,batch_size=mbatch_size,shuffle=False)
'''
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
'''
model1=Net1()
model1.to(device1)
# x_axes=[]
y_axes=[]
optimizer1=SGD(model1.parameters(),lr=mlearning_rate,momentum=mmomentum)
for _ in range(mepoch):
    for i,x1 in enumerate(data_loader_train):
        data1,label1=x1
        data1=data1.to(device1)
        label1=label1.to(device1)
        output1=model1(data1)
        loss1=F.cross_entropy(output1,label1)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        if(i%miternel==0):
            # torch.save(model1.state_dict(),'model1_state.pth')
            print(loss1.detach().data)
            y_axes.append(loss1.cpu().data)
torch.save(model1.state_dict(),'model1_state.pth')
x_axes=np.arange(len(y_axes))

plt.plot(x_axes,y_axes)
plt.legend('loss')
plt.show()


