import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.T = 440
        self.channels = 128
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 32), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(97, 32, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(32, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(32, 64, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(64, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # Layer 4
        self.padding3 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv4 = nn.Conv2d(64, 128, (8, 4))
        self.batchnorm4 = nn.BatchNorm2d(128, False)
        self.pooling4 = nn.MaxPool2d((2, 6))
        

        # Layer 5
        self.padding4 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv5 = nn.Conv2d(128, 256, (8, 4))
        self.batchnorm5 = nn.BatchNorm2d(256, False)
        self.pooling5 = nn.MaxPool2d((2, 4))
        
        # Layer 6
        self.padding5 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv6 = nn.Conv2d(256, 512, (8, 4))
        self.batchnorm6 = nn.BatchNorm2d(512, False)
        self.pooling6 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        self.fc1 = nn.Linear(512, 40)
        
    def forward(self, x):
        x = x.view(x.size(0),1,x.size(1),x.size(2))
        x = x.permute(0,1,3,2)
        
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # Layer 4
        x = self.padding3(x)
        x = F.elu(self.conv4(x))
        x = self.batchnorm4(x)
        x = F.dropout(x, 0.25)
        x = self.pooling4(x)

        # Layer 5
        #x = self.padding4(x)
        #x = F.elu(self.conv5(x))
        #x = self.batchnorm5(x)
        #x = F.dropout(x, 0.25)
        #x = self.pooling5(x)

        # Layer 6
        #x = self.padding5(x)
        #x = F.elu(self.conv6(x))
        #x = self.batchnorm6(x)
        #x = F.dropout(x, 0.25)
        #x = self.pooling6(x)

        # FC Layer
        x = x.view(-1, 512)
        x = torch.sigmoid(self.fc1(x))
        return x