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
from mData_Loader import data_loader_train1
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize
import matplotlib.pyplot as plt
import random
from mAttention import Attention1





model_loaded = Attention1(1,1)# size depend on data shape
for para1 in model_loaded.parameters():
    print(para1)
# model_loaded.load_state_dict(torch.load('./model1_state.pth'))
# model_loaded.eval()
# with torch.no_grad():
#     dataset_test = MNIST('./data', train=True, transform=
#     Compose([ToTensor(), Normalize(0.1, 0, 3)]), )
#     index1 = random.randint(1, 100)
#     data1, _ = dataset_test.test_data[index1], dataset_test.test_labels[index1]
#     x_test = data1
#     data1 = data1.float()
#     data_a, data_b = data1[0:14, 0:14].reshape(1, -1), data1[0:14, 14:28].reshape(1, -1)
#     data_c, data_d = data1[14:28, 0:14].reshape(1, -1), data1[14:28, 14:28].reshape(1, -1)
#     data1 = torch.cat([data_a, data_b, data_c, data_d], dim=0)
#     output1 = model_loaded(data1)
#     output1 = torch.cumsum(output1,dim=0)[3,:]
#
#     plt.imshow(x_test.numpy(), cmap='gray')
#     _, indices = torch.max(output1, dim=0)
#     plt.xlabel('predition is ' + str(indices.item()))
#     plt.show()
