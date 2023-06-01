import torch
from modulex import Net1
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize
import matplotlib.pyplot as plt
import random


# device1=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# x1=torch.load('./model1_state.pth')

model_loaded=Net1()
model_loaded.load_state_dict(torch.load('./model1_state.pth'))
model_loaded.eval()
with torch.no_grad():
    dataset_test = MNIST('./data', train=True, transform=
    Compose([ToTensor(), Normalize(0.1, 0, 3)]),)
    index1=random.randint(1,100)
    x_test,_=dataset_test.test_data[index1],dataset_test.test_labels[index1]
    x_test=x_test.float()
    output1=model_loaded(x_test.view(1,28,28))
    plt.imshow(x_test.numpy(),cmap='gray')
    _,indices=torch.max(output1,dim=1)
    plt.xlabel('predition is '+str(indices.item()))
    plt.show()

