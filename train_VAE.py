import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os, time, tqdm
import argparse
from conv_vae import VAE
from losses import new_vae_loss
from network_utils import EarlyStop, binary, normalise
from torch.utils.data import DataLoader
from custom_dataloader.custom_elv import customDataset
from custom_dataloader.augmentations import RotateTransform
from torchsummary import summary

def main(args):
    DATASET = args.dataset
    summary_mode = args.summary

    #################
    ### Load Data ###
    #################

    if DATASET == 'MNIST':

        mnist_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: binary(x))
        ])

        train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=mnist_transform)
        train_iter = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=torch.get_num_threads())
    elif DATASET == 'custom':
        processed_path = './data/test'
        train_iter = DataLoader(customDataset(processed_path, transform=RotateTransform([0, 90, 180, 270])), batch_size = 32,
                                    shuffle = True, num_workers=torch.get_num_threads())

    ##################
    ### Load Model ###
    ##################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if DATASET == 'MNIST':
        net = VAE((1, 28, 28), nhid = 4)
    elif DATASET == 'custom':
        net = VAE((1, 32, 32), nhid = 256, elv=True)
    net.to(device)

    if summary_mode:
        summary(net, (1, 32, 32))
        exit()

    save_name = './weights/new_model/VAE.pt'


    ################
    ### Training ###
    ################

    lr = 1e-3
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)

    def adjust_lr(optimiser, decay_rate = 0.95):
        for param_group in optimiser.param_groups:
            param_group['lr'] *= decay_rate
    
    retrain = True
    if os.path.exists(save_name):
        print("Model parameters have already been trained. Retrain ? [y/n]")
        #ans = input()
        ans = 'y'
        if not (ans == 'y'):
            checkpoint = torch.load(save_name, map_location=device)
            net.load_state_dict(checkpoint["net"])
            optimiser.load_state_dict(checkpoint["optimiser"])
            for g in optimiser.param_groups:
                g['lr'] = lr

    
    early_stop = EarlyStop(patience = 50, save_name = save_name)
    net = net.to(device)
    
    max_epochs = args.nepochs
    print('Training on ', device)
    for epoch in range(max_epochs):
        train_loss, n , start = 0.0, 0, time.time()
        if DATASET == 'MNIST':
            for X, _ in tqdm.tqdm(train_iter, ncols = 50):
                X = X.to(device)
                X_hat, mean, logvar = net(X)

                l = new_vae_loss(X, X_hat, mean, logvar).to(device)
                optimiser.zero_grad()
                l.backward()
                optimiser.step()

                train_loss += l.cpu().item()
                n += X.shape[0]
            
            train_loss /= n

            print('epoch %d, train loss %.4f , time %.1f sec'
            % (epoch, train_loss, time.time() - start))
        
            adjust_lr(optimiser)
            
            if (early_stop(train_loss, net, optimiser)):
                break
        elif DATASET == 'custom':
            for batch in tqdm.tqdm(train_iter, ncols = 50):
                im0 = normalise(batch['image_0'].to(device))
                im1 = normalise(batch['image_1'].to(device))
                print('Input shape: ', im0.shape)
                print('Target shape:', im1.shape)
                im1_hat, mean, logvar = net(im0)
                print('Out shape: ', im1_hat.shape)

                l = new_vae_loss(im1, im1_hat, mean, logvar).to(device)
                optimiser.zero_grad()
                l.backward()
                optimiser.step()

                train_loss += l.cpu().item()
                n += im0.shape[0]
            
            train_loss /= n

            print('epoch %d, train loss %.4f , time %.1f sec'
            % (epoch, train_loss, time.time() - start))
        
            adjust_lr(optimiser)
            
            if (early_stop(train_loss, net, optimiser)):
                break

    checkpoint = torch.load(early_stop.save_name)
    net.load_state_dict(checkpoint["net"])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--nepochs', type = int, default = 100)
    parser.add_argument('--summary', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    main(args)