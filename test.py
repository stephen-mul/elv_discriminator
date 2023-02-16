### Imports

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

from conv_vae import simple_VAE, VAE
from cond_vae import simple_cVAE

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
    if MODE == 'simple_VAE':
        net = simple_VAE().to(device)
        net.load_state_dict(torch.load('/home/stephen/notgan_workdir/vae/weights/simple_VAE/vae.pth'))
    elif MODE == 'simple_cVAE':
        net = simple_cVAE().to(device)
        net.load_state_dict(torch.load('/home/stephen/notgan_workdir/vae/weights/simple_cVAE/cvae.pth'))
    elif MODE == "VAE":
        net = VAE((1, 28, 28), nhid = 4).to(device)
        net.load_state_dict(torch.load('/home/stephen/notgan_workdir/elv_vae/weights/new_model/VAE.pt')['net'])
    else:
        print('Invalid network mode. Must be either simple_VAE, simple_cVAE, or VAE')
    

    net.eval()
    if MODE == 'simple_VAE' or MODE == 'VAE':
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
                print(np.squeeze(outimg).shape)
                plt.subplot(122)
                plt.imshow(np.squeeze(outimg))
                plt.savefig(f'/home/stephen/notgan_workdir/vae/plots/test_plot_{test_count}.png')
                np.save(f'/home/stephen/notgan_workdir/vae/plots/test_plot_{test_count}.npy', np.squeeze(outimg))
                test_count += 1
                if test_count == 10:
                    break
    elif MODE == 'simple_cVAE':
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