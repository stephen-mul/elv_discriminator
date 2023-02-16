### The following is an import of PyTorch libraries.

import torch
import torch.nn as nn
import torch.nn.functional as F
from network_blocks import Encoder, Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################
### Simple Conditional Variational Autoencoder Class ###
########################################################

class simple_VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim = 32*20*20, zDim=256):
        super(simple_VAE, self).__init__()
        self.featureDim = featureDim

        # Initilaising 2 convolutional layers and 2 fully-connected layers for encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initiliasing full-connecte layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.Conv2d(32, 16, 5, padding=4)
        self.decConv2 = nn.Conv2d(16, imgChannels, 5, padding=4)

    def encoder(self, x):

        # Input fed into 2 conv layers sequentially
        # Output feauter maps fed into fully connected layers at same time
        # Mu and logVar are used for generating middle representation z and KL divergence
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        print(x.shape)
        x = x.view(-1, self.featureDim)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterise(self, mu, logVar):
        # Reparameterisation takes in input mu and logvar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z fed back into full-connected layer and then two conv layers
        # Generated output same size of original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):
        
        # Forward pass of model
        mu, logVar = self.encoder(x)
        z = self.reparameterise(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar
    

#################################################
### Conditional Variational Autoencoder Class ###
#################################################

class cVAE(nn.Module):
    def __init__(self, shape, nclass, nhid = 16, ncond =16):
        super(cVAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid, ncond = ncond)
        self.decoder = Decoder(shape, nhid, ncond = ncond)
        self.label_embedding = nn.Embedding(nclass, ncond)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y):
        y = self.label_embedding(y)
        mean, logvar = self.encoder(x, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar

    def generate(self, class_idx):
        if (type(class_idx) is int):
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(device)
        if (len(class_idx.shape) == 0):
            batch_size = None
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        y = self.label_embeddings(class_idx)
        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res


