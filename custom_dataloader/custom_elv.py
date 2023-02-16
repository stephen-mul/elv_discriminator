import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.all_filenames = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        file = np.load(os.path.join(self.path, selected_filename))
        #print(file.files)
        sample = {}
        for entry in file.files:
            sample[entry] = torch.from_numpy(file[entry]).unsqueeze(0)
            #print(type(sample[entry]))

        if self.transform:
            sample = self.transform(sample)

        #print(sample)

        return sample
