### Based on https://github.com/chendaichao/VAE-pytorch/blob/master/Models/VAE/model.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)
    
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

### Writing a convolutional block to be used for both Encoder and Decoder

### View class is necessary - from https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/12

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class ConvBlock(nn.Module):
    def __init__(self, shape, nhid=16, ncond=0, encoder=True):
        super(ConvBlock, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        if encoder:
            self._conv_block = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0), nn.BatchNorm2d(16), nn.ReLU(inplace = True),
                                        nn.Conv2d(16, 32, 5, padding = 0), nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                                        nn.MaxPool2d(2, 2),
                                        nn.Conv2d(32, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                        nn.Conv2d(64, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                        nn.MaxPool2d(2, 2),
                                        Flatten(), MLP([ww*hh*64, 256])
                                        )
            self.conv_block = nn.Sequential(nn.Conv2d(c, 16, 3, padding = 1, stride = 2), nn.BatchNorm2d(16), nn.ReLU(inplace = True),
                                        nn.Conv2d(16, 32, 3, padding = 1, stride = 2), nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                                        nn.MaxPool2d(2, 2),
                                        Flatten(), MLP([ww*hh*32, 256])
                                        )
        else:
            ## decoder block here
            ## block that runs
            self._conv_block = nn.Sequential(MLP([nhid+ncond, 1024]),
                                        nn.Unflatten(1, (64, 4, 4)),
                                        nn.Conv2d(64,64*4, 3, 1, 1), nn.BatchNorm2d(64*4), nn.ReLU(inplace=True), nn.PixelShuffle(2),
                                        nn.MaxPool2d(3, 1, 1, 1),
                                        nn.Conv2d(64,64*4, 3, 1, 1), nn.BatchNorm2d(64*4), nn.ReLU(inplace=True), nn.PixelShuffle(2),
                                        nn.MaxPool2d(3, 1, 1, 1),
                                        nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), 
                                        nn.MaxPool2d(5, 1, 0, 1),
                                        nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                        nn.MaxPool2d(5, 1, 0, 1), 
                                        nn.Sigmoid()
                                        )
            
            ### Small block
            self.__conv_block = nn.Sequential(MLP([nhid+ncond,  4096]),
                                        nn.Unflatten(1, (64, 8, 8)),
                                        nn.Conv2d(64,64*64, 3, 1, 1), nn.BatchNorm2d(64*64), nn.ReLU(inplace=True), nn.PixelShuffle(8),
                                        nn.MaxPool2d(3, 1, 1, 1),
                                        nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                        nn.MaxPool2d(3, 1, 1, 1),
                                        nn.Conv2d(32, 1, 3, 1, 1), nn.BatchNorm2d(1),
                                        nn.Sigmoid()
                                        )
            ### Resize convolution block https://distill.pub/2016/deconv-checkerboard/
            self.conv_block = nn.Sequential(MLP([nhid+ncond]),
                                        nn.Unflatten(1, (32, 4, 4)),
                                        nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                        nn.MaxPool2d(3, 1, 1, 1),
                                        Interpolate((32, 32), mode='nearest'),
                                        nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                        nn.MaxPool2d(3, 1, 1, 1),
                                        Interpolate((64, 64), mode='nearest'),
                                        nn.Conv2d(32, 1, 3, 1, 1), nn.BatchNorm2d(1),
                                        nn.Sigmoid()
                                        )

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        test_block = 'on'
        if test_block =='on':
            print('Encoder test block on')
            self.encode = ConvBlock(shape, nhid, ncond, encoder=True)
        else:
            print('Encoder test block off')
            self.encode = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0), nn.BatchNorm2d(16), nn.ReLU(inplace = True),
                                        nn.Conv2d(16, 32, 5, padding = 0), nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                                        nn.MaxPool2d(2, 2),
                                        nn.Conv2d(32, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                        nn.Conv2d(64, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                        nn.MaxPool2d(2, 2),
                                        Flatten(), MLP([ww*hh*64, 256, 128])
                                        )
        #self.calc_mean = MLP([128+ncond, 64+nhid, nhid], last_activation = False)
        #self.calc_logVar = MLP([128+ncond, 64+nhid, nhid], last_activation = False)
        self.calc_mean = MLP([256+ncond, nhid], last_activation=False)
        self.calc_logVar = MLP([256+ncond, nhid],last_activation=False)
    def forward(self, x, y = None):
        x = self.encode(x)
        if (y is None):
            return self.calc_mean(x), self.calc_logVar(x)
        else:
            return self. calc_mean(torch.cat((x, y), dim = 1), self.calc_logVar(torch.cat(x, y), dim = 1))

class Decoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        test_block = 'on'
        if test_block == 'on':
            print('Decoder test block on')
            self.decode = ConvBlock(shape, nhid, ncond, encoder=False)
        else:
            print('Decoder test block off')
            self.decode = nn.Sequential(MLP([nhid+ncond, nhid+64, nhid+128, nhid+256, c*w*h], last_activation = False), nn.Sigmoid())
    def forward(self, z, y = None):
        c, w, h = self.shape
        if (y is None):
            return self.decode(z).view(-1, c, w, h)
        else:
            return self.decode(torch.cat((z,y), dim = 1)).view(-1, c, w, h)


