import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from raw_elv_dataset import rawElvDataset
from transforms import multiBilinearDownsample

def save_levels(levels_from_batch):
    print('Saving')
    print(levels_from_batch.keys())
    file_num = levels_from_batch['file_num']
    print(type(levels_from_batch))
    exit()
    del levels_from_batch['img_idx']
    del levels_from_batch['file_num']
    #np.savez(f'../data/test/processed_elv_{file_num}.npz', **levels_from_batch)



def main(args):
    ###########################
    ### Creating Out Folder ###
    ###########################
    RAW = args.raw_dir_path
    OUT_NAME = args.out_dir_name
    out_dir_path = os.path.join('../data/', OUT_NAME)

    ### Check if folder exists - create if not
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    ######################
    ### Create Dataset ###
    ######################

    elv_dataset = rawElvDataset(RAW, out_dir_path, multiBilinearDownsample())

    ###################
    ### Data Loader ###
    ###################
    
    dataloader = DataLoader(elv_dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(f'Batch number {i_batch}')
        print(f'Batch shape {len(sample_batched)}')
        #save_levels(sample_batched)





if __name__ == "__main__":
    print('running')
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir_path", type=str, required=True)
    parser.add_argument("--out_dir_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
