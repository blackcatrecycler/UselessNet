import torch
from torch import nn
import torch.nn.functional as F

class NMSLBlock(nn.Module):
    def __init__(self,in_channels):
        super(NMSLBlock, self).__init__()
        self.c1=nn.Conv2d(in_channels,in_channels,5,1,2)
        self.pool=nn.MaxPool2d(3,stride=1,padding=1)

    def forward(self, x):
        x=self.c1(x)
        x=F.relu(x)
        x=self.pool(x)
        return x



class SBNet(nn.Module):
    def __init__(self,num,inp):
        super(SBNet, self).__init__()
        self.cnm=nn.Conv2d(3,inp,5,1,2)
        self.nmsls=self.make_all_layer(num,inp)
        self.final=nn.Linear(inp*100*100,10)

    def make_layer(self,n,inp):
        layer=nn.Sequential()
        for i in range(n):
            layer.add_module('n{}'.format(n),NMSLBlock(inp))
        return layer
    def make_all_layer(self,n,inp):
        layer=nn.Sequential()
        for i in range(n):
            layer.add_module('n{}'.format(n),self.make_layer(n,inp))
        return layer


    def forward(self, x):
        x=self.cnm(x)
        x=self.nmsls(x)
        x=x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x=self.final(x)
        return x
