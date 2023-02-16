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