### The following is an import of PyTorch libraries.

import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################
### Convolutional Variational Autoencoder Class ###
###################################################

class cVAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim = 32*20*20, zDim=256, ncond=16, nclass=10):
        super(cVAE, self).__init__()

        # Initilaising 2 convolutional layers and 2 fully-connected layers for encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim+ncond, zDim)
        self.encFC2 = nn.Linear(featureDim+ncond, zDim)

        # Initiliasing full-connecte layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim+ncond, featureDim)
        self.decConv1 = nn.Conv2d(32, 16, 5, padding=4)
        self.decConv2 = nn.Conv2d(16, imgChannels, 5, padding=4)

        # Embedding function
        self.label_embedding = nn.Embedding(nclass, ncond)

    def encoder(self, x, y):

        # Input fed into 2 conv layers sequentially
        # Output feauter maps fed into fully connected layers at same time
        # Mu and logVar are used for generating middle representation z and KL divergence
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32*20*20)
        mu = self.encFC1(torch.cat((x, y), dim=1)) # Concatentating embedded y for conditioning
        logVar = self.encFC2(torch.cat((x, y), dim=1))
        return mu, logVar

    def reparameterise(self, mu, logVar):
        # Reparameterisation takes in input mu and logvar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z, y):

        # z fed back into full-connected layer and then two conv layers
        # Generated output same size of original input
        x = F.relu(self.decFC1(torch.cat((z, y), dim=1)))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x, y):
        
        # Forward pass of model
        y = self.label_embedding(y)
        mu, logVar = self.encoder(x, y)
        z = self.reparameterise(mu, logVar)
        out = self.decoder(z,y)
        return out, mu, logVar
    



