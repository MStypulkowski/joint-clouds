import os
from tarfile import ExFileObject
from turtle import forward

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

#from models.scale_blocks import H_Block, Z_Block
from models.backbones import CMLP, SMLP, FMLP
from models.pointnet import PointNet, MLP

from utils import count_trainable_parameters

class VanillaVAE(nn.Module):

    def __init__(self, x_dim, y_dim, h_dim=64, e_dim = 64, z_dim=64, n_layers=3):
        super(VanillaVAE, self).__init__()
        self.y_dim =y_dim
        self.h_dim=h_dim
        self.e_dim=e_dim
        self.z_dim=z_dim
        self.n_layers= n_layers

        # default option for PointNet model is not to perform pooling and fully conected layers
        self.pn_xh = PointNet(x_dim, h_dim, n_layers=n_layers)   #(x * h_dim)
        self.pn_hz1 = PointNet(h_dim, 2 * z_dim, n_layers=n_layers) 
        self.pn_z1x = PointNet(z_dim, 2 * x_dim, n_layers=n_layers)

        self.mlp_he = MLP(h_dim, e_dim, n_layers=n_layers)
        self.mlp_ey = MLP(e_dim, y_dim, n_layers=n_layers)
        
        #self.mlp_he = MLP(in_dim=x_dim, out_dim=h_2, hid_dim=h_dim,  n_resnet_blocks=n_layers)
        #self.mlp_ey = MLP(in_dim=h_2, out_dim=y_dim, hid_dim=h_dim, n_resnet_blocks=n_layers)
        #self.mlp_ez2 = MLP(h_2, 2 * z2_dim, n_layers=n_layers)

        self.Encoder = Encoder(input_dim=self.h_dim, hidden_dim=self.e_dim, latent_dim=self.z_dim)
        self.Decoder = Decoder(latent_dim=self.z_dim, hidden_dim=self.e_dim, output_dim=10)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std 

    def forward(self, x):
    
        print('x', x.shape)

        #pointNet on input
        h = self.pn_xh(x)
        print('h', h.shape)
        h = self.Encoder(h) #(h_dim, z_dim)
        print(h)
        e = torch.max(h, 2, keepdim=True)[0]
        e = e.squeeze(-1)
        # print('h_pooled', e.shape)
        e = self.mlp_he(e)
        # print('e', e.shape)

        #classification
        logits = self.mlp_ey(e)
        # print('logits', logits.shape)

        
        mean, log_var = self.mlp_he(e).chunk(2, 1)
        #mean, log_var = self.Encoder(x)
        z = self.reparametrization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        kl_z = 0.5 * (mean**2)
        # kl_z = 0.5 * (mean**2 + torch.exp(log_var) - 1).sum([1, 2])
        print('kl_z', kl_z.shape)

        # log-likelihood
        nll = 0.5 * (np.log(2. * np.pi) + log_var )
        # nll = 0.5 * (np.log(2. * np.pi) + log_var + torch.exp(-log_var) * (x - mean)**2).sum([1, 2])

        print('nll', nll.shape)

        # final ELBO
        elbo = (nll  + kl_z).mean()
        return logits, elbo, x_hat

    def sample(self, n_samples, n_points, device='cuda'):
        z = torch.randn(n_samples, self.z_dim, n_points).to(device)
        print('z', z.shape)
        mu_x, log_var_x = self.pn_z1x(z).chunk(2, 1)
        print('params', mu_x.shape, log_var_x.shape)
        
        
        x = self.reparametrization(mu_x, log_var_x)
        print('z', x.shape)
        
        # print('params_x', mu_x.shape, logvar_x.shape)
        
        return x

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.input_dim=input_dim
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

    model = VanillaVAE(2, 4, h_dim=32, e_dim=64, z_dim=128, n_layers=3).to(device)
    print('Number of trainable parameters:', count_trainable_parameters(model))
    print(model)

    x = torch.randn(8, 2, 16).to(device)
    loss = model(x)
    print(loss.item())

    x_gen = model.sample(10, 100)
    print(x_gen.shape)



