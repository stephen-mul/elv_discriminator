### Code based on example at https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71 

### Imports

import torch
from torchvision import datasets, transforms
import argparse
from torch.utils.data import DataLoader

from conv_vae import VAE
from cond_vae import cVAE
from custom_dataloader.custom_elv import customDataset
from losses import vae_loss

def normalise(im):
    min_val = torch.min(im)
    im -= min_val
    max_val = torch.max(im)
    im = im/max_val
    return im


def main(args):


    MODE = args.mode
    DATASET = args.dataset

    ##########################
    ### Main Training Loop ###
    ##########################

    ########################
    ### GPU Availability ###
    ########################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##################
    ### Dataloader ###
    ##################

    if DATASET == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
            transform=transforms.ToTensor()), batch_size = 128, shuffle=True
        )
    elif DATASET == 'custom':
        processed_path = '/home/stephen/notgan_workdir/vae/data/test'
        train_loader = DataLoader(customDataset(processed_path), batch_size = 16,
                                    shuffle = True, num_workers=4)

    #############################
    ### Network and Optimiser ###
    #############################

    if MODE == 'VAE':
        net = VAE(featureDim=73728).to(device)
    elif MODE == 'cVAE':
        net = cVAE().to(device)
    else:
        print('Invalid network mode. Must be either VAE or cVAE')
    
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)

    ################
    ### Training ###
    ################

    if MODE == 'VAE':
        for epoch in range(1):
            for idx, data in enumerate(train_loader, 0):
                im0 = normalise(data['image_0'].to(device))
                im1 = normalise(data['image_1'].to(device))

                # Pass batch to network
                out, mu, logVar = net(im0)

                # Calculate loss
                loss = vae_loss(out, im1, mu, logVar)

                # Backprop
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            print(f'Epoch {epoch}: Loss {loss}')

        print('Saving Model')
        torch.save(net.state_dict(), '/home/stephen/notgan_workdir/vae/weights/VAE/vae.pth')
    elif MODE == 'cVAE':

        for epoch in range(1):
            for idx, data in enumerate(train_loader, 0):
                imgs, y = data
                imgs = imgs.to(device)
                y = y.to(device)

                # Pass batch to network
                out, mu, logVar = net(imgs, y)

                # Calculate loss
                loss = vae_loss(out, imgs, mu, logVar)

                # Backprop
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            print(f'Epoch {epoch}: Loss {loss}')

        print('Saving Model')
        torch.save(net.state_dict(), '/home/stephen/notgan_workdir/vae/weights/cVAE/cvae.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--dataset', default='MNIST')
    args = parser.parse_args()
    main(args)
