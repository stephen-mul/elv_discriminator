### Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

####################
### Model Losses ###
####################

def kl_div(mu, logVar):
    return 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())

def bin_cross_entropy(out, imgs):
    return F.binary_cross_entropy(out, imgs, size_average=False)

def vae_loss(out, imgs, mu, logVar):
    return bin_cross_entropy(out, imgs) + kl_div(mu, logVar)

### loss for updated VAEs

BCE_loss = nn.BCELoss(reduction = "sum")
def new_vae_loss(X, X_hat, mean, logvar):
    reconstruction_loss = BCE_loss(X_hat, X)
    KL_div = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_div