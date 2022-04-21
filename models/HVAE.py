import os
from turtle import forward

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import PointNet, MLP
from utils import count_trainable_parameters

class HVAE(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim=64, h_2= 64, z1_dim=64, n_layers=3):
        super(VanillaVAE, self).__init__()

        # default option for PointNet model is not to perform pooling and fully conected layers
        self.pn_xh = PointNet(x_dim, h_dim, n_layers=n_layers)
        self.pn_hz1 = PointNet(h_dim, 2 * z1_dim, n_layers=n_layers)
        self.pn_z1x = PointNet(z1_dim, 2 * x_dim, n_layers=n_layers)

        self.mlp_he = MLP(h_dim, h_2, n_layers=n_layers)
        self.mlp_ey = MLP(h_2, y_dim, n_layers=n_layers)
        #self.mlp_ez2 = MLP(h_2, 2 * z2_dim, n_layers=n_layers)

        self.Encoder = Encoder(input_dim= 64, hidden_dim=128, latent_dim=64)
        self.Decoder = Decoder(latent_dim=64, hidden_dim=128, output_dim=10)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

   
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var


class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        e_h      = self.LeakyReLU(self.FC_input(x))
        e_h      = self.LeakyReLU(self.FC_input2(e_h))
        mean     = self.FC_mean(e_h)
        log_var  = self.FC_var(e_h)                           
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

    def forward(self, x, only_classify= False):
        print('x', x.shape)
        h= self.pn_xh(x)
        print('h', h.shape)
        mean, log_var = self.Encoder(h)

        z1 = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)

        x_hat= self.Decoder(z1)

        kl_z1 = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        print('kl_z1', kl_z1.shape)
        
        # log-likelihood
        nll = 0
        print('nll', nll.shape)
        
        # final ELBO
        elbo = (kl_z1).mean()

        return elbo
        
        


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hvae = hVAE(2, 4, h_dim=32, e_dim=64, z1_dim=128, z2_dim=256).to(device)
    print('Number of trainable parameters:', count_trainable_parameters(hvae))

    x = torch.randn(8, 2, 16).to(device)
    loss, logits = hvae(x)
    print(loss.item(), logits.shape)

    x_gen = hvae.sample(10, 100)
    print(x_gen.shape)
