from torch import  nn
import torch.nn.functional as F
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,5,5)
        self.pl1=nn.MaxPool2d((2,2))
        self.fc1=nn.Linear(80,100)
        self.fc2=nn.Linear(100,10)

    def forward(self,input1):
        x=self.conv1(input1)
        x=self.pl1(x)
        x=self.conv2(x)
        x=self.pl1(x)
        x=nn.functional.relu(x)
        x=x.view(-1,80)
        x=self.fc1(x)
        x=nn.functional.relu(x)
        x=self.fc2(x)
        return  x
        # return F.log_softmax(x)




