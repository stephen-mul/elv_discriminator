### Imports

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

from conv_vae import VAE
from cond_vae import cVAE

def main(args):

    MODE = args.mode
    DATASET = args.dataset

    ########################
    ### GPU Availability ###
    ########################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##################
    ### Dataloader ###
    ##################

    if DATASET == 'MNIST':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False,
            transform=transforms.ToTensor()), batch_size=1
        )

    print('Loading Model')
    if MODE == 'VAE':
        net = VAE().to(device)
        net.load_state_dict(torch.load('/home/stephen/notgan_workdir/vae/weights/VAE/vae.pth'))
    elif MODE == 'cVAE':
        net = cVAE().to(device)
        net.load_state_dict(torch.load('/home/stephen/notgan_workdir/vae/weights/cVAE/cvae.pth'))
    else:
        print('Invalid network mode. Must be either VAE or cVAE')
    

    net.eval()
    if MODE == 'VAE':
        with torch.no_grad():
            test_count = 0
            for data in random.sample(list(test_loader), 10):
                imgs, _ = data
                imgs = imgs.to(device)
                img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
                plt.subplot(121)
                plt.imshow(np.squeeze(img))
                out, _, _ = net(imgs)
                outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
                plt.subplot(122)
                plt.imshow(np.squeeze(outimg))
                plt.savefig(f'/home/stephen/notgan_workdir/vae/plots/test_plot_{test_count}.png')
                test_count += 1
                if test_count == 10:
                    break
    elif MODE == 'cVAE':
        with torch.no_grad():
            test_count = 0
            for data in random.sample(list(test_loader), 10):
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)
                img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
                plt.subplot(121)
                plt.imshow(np.squeeze(img))
                out, _, _ = net(imgs, labels)
                outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
                plt.subplot(122)
                plt.imshow(np.squeeze(outimg))
                plt.savefig(f'/home/stephen/notgan_workdir/vae/plots/test_plot_{test_count}.png')
                test_count += 1
                if test_count == 10:
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--dataset', default='MNIST')
    args = parser.parse_args()
    main(args)