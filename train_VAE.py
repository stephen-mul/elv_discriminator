import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os, time, tqdm
import argparse
from conv_vae import VAE
from losses import new_vae_loss
from network_utils import EarlyStop, binary
from torch.utils.data import DataLoader

def main(args):
    DATASET = args.dataset

    #################
    ### Load Data ###
    #################

    if DATASET == 'MNIST':

        mnist_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: binary(x))
        ])

        train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=mnist_transform)
        train_iter = DataLoader(train_data, batch_size=512, shuffle=True)

    ##################
    ### Load Model ###
    ##################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VAE((1, 28, 28), nhid = 4)
    net.to(device)
    #print(net)
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
        ans = input()
        if not (ans == 'y'):
            checkpoint = torch.load(save_name, map_location=device)
            net.load_state_dict(checkpoint["net"])
            optimiser.load_state_dict(checkpoint["optimiser"])
            for g in optimiser.param_groups:
                g['lr'] = lr

    max_epochs = 1000
    early_stop = EarlyStop(patience = 20, save_name = save_name)
    net = net.to(device)

    print('Training on ', device)
    for epoch in range(max_epochs):
        train_loss, n , start = 0.0, 0, time.time()
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

    checkpoint = torch.load(early_stop.save_name)
    net.load_state_dict(checkpoint["net"])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST')
    args = parser.parse_args()
    main(args)