import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset

class rawElvDataset(Dataset):
    def __init__(self, raw_data_dir, out_data_dir=None, transform=None):
        self.raw_data_dir = raw_data_dir
        self.out_data_dir = out_data_dir
        self.all_filenames = os.listdir(self.raw_data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        regex = re.compile(r'\d+')
        filenum = regex.findall(selected_filename)
        image = np.load(os.path.join(self.raw_data_dir,selected_filename))['data'][:,:,0]
        image = torch.from_numpy(image)
        sample = {'data': image,
                    'img_idx': idx,
                    'file_num': filenum
                    }

        if self.transform:
            sample = self.transform(sample, self.out_data_dir)
        
        return sample

