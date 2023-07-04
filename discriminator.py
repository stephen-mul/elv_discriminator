### Necesary imports ###

import torch
import torch.nn as nn
import torch.nn.functional as F

###################################
### Simple discriminator module ###
###################################

class discriminator(nn.Module):
    def __init__(self, img_channels=1, n_classes=10):
        super(discriminator, self).__init__()
        ### Initialise one hot enconder ##
        
        ### Initialise layers ###
        self.Conv1 = nn.Conv2d(img_channels, 16, 3)
        self.FC1 = nn.Linear(16*26*26, n_classes)

    def forward(self, image):
        x = self.Conv1(image)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        output = F.log_softmax(x)

        return output



