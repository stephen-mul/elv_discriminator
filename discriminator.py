### Necesary imports ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import one_hot_encoder

###################################
### Simple discriminator module ###
###################################

class discriminator(nn.Module):
    def __init__(self, ):