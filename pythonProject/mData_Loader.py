import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize
from mConfig import Config1
config1=Config1()
dataset_train1=MNIST('./data',train=True,transform=
               Compose([ToTensor(),Normalize(0.1307,0.3081)]),
                download = True
)

data_loader_train1 = DataLoader(dataset=dataset_train1,batch_size=config1.BATCH_SIZE,shuffle=True,drop_last=True)
