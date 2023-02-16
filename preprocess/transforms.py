import torch.nn.functional as F
import numpy as np
import os

class multiBilinearDownsample(object):
    def __init__(self, nlevels=2, lowest_dim=32):
        self.nlevels = nlevels 
        self.lowest_dim = lowest_dim # dimension of lowest level - default 32

    def __call__(self, sample, out_data_dir):
        image, idx , file_num= sample['data'], sample['img_idx'], sample['file_num']
        image = image.float()
        out = {}
        
        for n in np.arange(self.nlevels):
            out_dim = self.lowest_dim*(n+1)
            down = F.interpolate(image.unsqueeze(0).unsqueeze(1), size=out_dim, mode = 'bilinear', align_corners=True)
            out[f'image_{n}'] = down.squeeze(1).squeeze(0).numpy()

        out['img_idx'] = idx
        out['file_num'] = file_num

        if out_data_dir:
            del out['img_idx']
            del out['file_num']
            np.savez(os.path.join(out_data_dir,f'processed_elv_{file_num[0]}.npz'), **out)


        return out
