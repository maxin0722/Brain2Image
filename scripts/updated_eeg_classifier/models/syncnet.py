import torch.nn.functional as F
import torch
import scipy.io
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import numpy as np

class Model(nn.Module):
    def __init__(self, num_channels=128, input_size=440, num_classes=40, num_filters=128, filter_width = 40, pool_size=40):
        super(Model, self).__init__()
        self.num_filters = num_filters 
        self.num_channels = num_channels
        self.filter_width = filter_width
        
        self.b = nn.Parameter(torch.FloatTensor(1,1,num_channels,num_filters).uniform_(-0.05, 0.05))
        self.bias = nn.Parameter(torch.FloatTensor(num_filters).fill_(0))
        self.omega = nn.Parameter(torch.FloatTensor(1,1,1,num_filters).uniform_(0, 1))
        self.cl_Wy = int(np.ceil(float(input_size)/float(pool_size)) * num_filters)
        
        if(filter_width%2 == 0):
            self.t = nn.Parameter(torch.FloatTensor(np.reshape(range(-int(filter_width/2),int(filter_width/2)),[1,int(filter_width),1,1])).fill_(0))
        else:
            self.t = nn.Parameter(torch.FloatTensor(np.reshape(range(-int((filter_width-1)/2),int((filter_width-1)/2) + 1),[1,int(filter_width) ,1,1])).fill_(0))
        
        self.phi_ini = nn.Parameter(torch.FloatTensor(1,1,num_channels, num_filters).normal_(0,0.05))
        self.beta = nn.Parameter(torch.FloatTensor(1,1,1,num_filters).uniform_(0, 0.05))
        
        ## Only stride and dilation values of 1 are supported. If you use different values, padding values wont be correct
        P = ((input_size-1)-input_size + (filter_width-1))
        if(P%2 == 0):
            self.padding = (P//2, P//2 + 1)
        else:
            self.padding = (P//2, P//2)
        
        self.pool = nn.MaxPool2d((1, pool_size), stride = (1, pool_size))
        self.classifier = nn.Linear(self.cl_Wy,num_classes)
    
    def forward(self,X):
        #X must be in the form of BxCx1xT or BxCxT
        X = X.permute(0,2,1)

        self.W_osc = torch.mul(self.b,torch.cos(self.t*self.omega + self.phi_ini))
        self.W_decay = torch.exp(-torch.pow(self.t,2)*self.beta)
        self.W = torch.mul(self.W_osc,self.W_decay)
        self.W = self.W.view(self.num_filters,self.num_channels,1,self.filter_width)
        if(len(X.size()) == 3):
            X = X.unsqueeze(2)
    
        res = F.conv2d(F.pad(X,self.padding,"constant", 0).float(),self.W.float(),bias = self.bias,stride=1 )
        res = self.pool(F.pad(res,self.padding, "constant", 0))
        res = res.view(-1,self.cl_Wy)
        self.beta = nn.Parameter(torch.clamp(self.beta, min=0))
        return self.classifier(F.relu(res)).squeeze()