from __future__ import absolute_import

from .vgg import *
from .preresnet import *
from .densenet import *
from .channel_selection import *

__all__ = ['vgg11', 'vgg13', 'vgg16']


def test_resnet(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:39:30 2018

@author: tshzzz
"""

cfg = {
       'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
       'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
       'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
       
       }

class VGG(nn.Module):
    
    def __init__(self,layers,num_class=10):
        super(VGG,self).__init__()
        
        self.img_channel = 3
        self.layers = layers
        
        self.conv = self.make_layers()
        self.fc = nn.Linear(512, num_class)
        
    def make_layers(self):
        
        layers = []
        in_channel = self.img_channel
        for idx in range(len(self.layers)):
            if self.layers[idx] == 'M':          
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channel,self.layers[idx],kernel_size=3,padding=1))
                layers.append(nn.BatchNorm2d(self.layers[idx]))
                layers.append(nn.ReLU())
                in_channel = self.layers[idx]
        return nn.Sequential(*layers)
            
    
    def forward(self,x):
        
        out = self.conv(x)
        out = out.view(-1,512)
        out = self.fc(out)
        
        return out
        

def vgg11():
    return VGG(cfg['vgg11'],num_class=10)

def vgg13():
    return VGG(cfg['vgg13'],num_class=10)
       
def vgg16():
    return VGG(cfg['vgg16'],num_class=10)

def test_vgg():
    import torchvision.models as models
    model = models.vgg16()
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            count += 1
    print(count)
    a = VGG16()
    count = 0
    for m in a.modules():
        if isinstance(m, nn.Conv2d):
            count += 1
    print(count)
    y = a(torch.randn(3,3,32,32))
    print(y.size())


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test_resnet(globals()[net_name]())
            print()
    
    test_vgg()