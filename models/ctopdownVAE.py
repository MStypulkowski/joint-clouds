import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from models.backbones import MLP, CMLP, PointNet, CPointNet
from utils import count_trainable_parameters


class ConditionalTopDownVAE(nn.Module):
    def __init__(self, x_dim, y_dim, h1_dim=64, h2_dim=64, e_dim=64, ze_dim=64, z1_dim=64, z2_dim=64, r1_dim=64, 
                encoder_hid_dim=64, decoder_hid_dim=64, encoder_n_layers=3, decoder_n_layers=3,
                activation='relu', use_batchnorms=False, use_lipschitz_norm=True, lipschitz_loss_weight=1e-6):
        super(ConditionalTopDownVAE, self).__init__()

        self.ze_dim = ze_dim
        self.z2_dim = z2_dim
        self.x_dim= x_dim
        self.z1_dim= z1_dim

        self.lipschitz_loss_weight = lipschitz_loss_weight

        # default option for PointNet model is not to perform pooling and fully conected layers
        self.pn_x_h1 = PointNet(x_dim, h1_dim, hid_dim=encoder_hid_dim, n_layers=encoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)
        self.pn_h1_h2 = PointNet(h1_dim, h2_dim, hid_dim=encoder_hid_dim, n_layers=encoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)
        self.pn_h2_z2 = PointNet(h2_dim, 2 * z2_dim, hid_dim=encoder_hid_dim, n_layers=encoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)
        self.cmlp_z2_r1= CMLP(z2_dim, r1_dim, ze_dim, hid_dim=decoder_hid_dim, n_layers=decoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)
        # self.cmlp_z2_z1= CMLP(z2_dim, 2 * z1_dim, ze_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        self.cmlp_r1_z1= CMLP(r1_dim, 2 * z1_dim, ze_dim, hid_dim=decoder_hid_dim, n_layers=decoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)
        self.pn_h1_z1 = PointNet(h1_dim, 2 * z1_dim, hid_dim=encoder_hid_dim, n_layers=encoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)
        self.cmlp_z1_x= CMLP(z1_dim + r1_dim, 2 * x_dim, ze_dim, hid_dim=decoder_hid_dim, n_layers=decoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)

        self.mlp_h2_e = MLP(h2_dim, e_dim, hid_dim=encoder_hid_dim, n_layers=encoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)
        self.mlp_e_ze = MLP(e_dim, 2 * ze_dim, hid_dim=encoder_hid_dim, n_layers=encoder_n_layers, activation=activation, last_activation=None, use_batchnorms=use_batchnorms, use_lipschitz_norm=use_lipschitz_norm)
        
    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, only_classify=False, epoch=None, save_dir=None):
        # bottom-up deterministic path
        h1 = self.pn_x_h1(x)
        delta_mu_z1, delta_logvar_z1 = self.pn_h1_z1(h1).chunk(2, 1)
        delta_mu_z1, delta_logvar_z1 = delta_mu_z1.permute(0, 2, 1).reshape(-1, self.z1_dim), delta_logvar_z1.permute(0, 2, 1).reshape(-1, self.z1_dim)

        h2 = self.pn_h1_h2(h1)
        delta_mu_z2, delta_logvar_z2 = self.pn_h2_z2(h2).chunk(2, 1)
        delta_mu_z2, delta_logvar_z2 = delta_mu_z2.permute(0, 2, 1).reshape(-1, self.z2_dim), delta_logvar_z2.permute(0, 2, 1).reshape(-1, self.z2_dim)

        e = torch.max(h2, 2, keepdim=True)[0]
        e = e.squeeze(-1)
        e = self.mlp_h2_e(e)
        delta_mu_ze, delta_logvar_ze = self.mlp_e_ze(e).chunk(2, 1)

        # top-down stochastic path
        ze = self.reparametrization(delta_mu_ze, delta_logvar_ze) # N x ze_dim
        ze = ze.unsqueeze(1).expand(-1, x.shape[-1], self.ze_dim).reshape(-1, self.ze_dim) # N * M x ze_dim but in correct order
        # if epoch is not None:
        #     plt.figure(figsize=(5, 10))
        #     ze_plot = ze.detach().cpu().numpy().reshape(-1)
        #     plt.hist(ze_plot, bins=20)
        #     plt.savefig(os.path.join(save_dir, 'figures/_ze.png'))

        z2= self.reparametrization(delta_mu_z2, delta_logvar_z2)
        # if epoch is not None:
        #     plt.figure(figsize=(5, 10))
        #     z2_plot = z2[:x.shape[2]].detach().cpu().numpy()
        #     plt.scatter(z2_plot[:, 0], z2_plot[:, 1])
        #     plt.savefig(os.path.join(save_dir, 'figures/_z2.png'))

        # skip-connection
        r1 = self.cmlp_z2_r1(z2, ze)
        mu_z1, logvar_z1 = self.cmlp_r1_z1(r1, ze).chunk(2, 1)
        # mu_z1, logvar_z1 = self.cmlp_z2_z1(z2, ze).chunk(2, 1)
        z1 = self.reparametrization(mu_z1 + delta_mu_z1, logvar_z1 + delta_logvar_z1)
        
        # if epoch is not None:
        #     plt.figure(figsize=(5, 10))
        #     z1_plot = z1[:x.shape[2]].detach().cpu().numpy()
        #     plt.scatter(z1_plot[:, 0], z1_plot[:, 1])
        #     plt.savefig(os.path.join(save_dir, 'figures/_z1.png'))
        
        mu_x, logvar_x = self.cmlp_z1_x(torch.cat([z1, r1], dim=1), ze).chunk(2, 1) # TODO check if z1 + r1, cat[z1, r1] or cat[z1, mu_z1]
        mu_x = mu_x.reshape(-1, x.shape[-1], self.x_dim).permute(0, 2, 1)
        logvar_x = logvar_x.reshape(-1, x.shape[-1], self.x_dim).permute(0, 2, 1)
        x_recon = self.reparametrization(mu_x, logvar_x)
        
        # ELBO
        # KL
        kl_ze = 0.5 * (delta_mu_ze**2 + torch.exp(delta_logvar_ze) - delta_logvar_ze - 1).sum()
        kl_z2 = 0.5 * (delta_mu_z2**2 + torch.exp(delta_logvar_z2) - delta_logvar_z2 - 1).sum()
        kl_z1 = 0.5 * (delta_mu_z1**2 * torch.exp(-logvar_z1) + torch.exp(delta_logvar_z1) - delta_logvar_z1 - 1).sum()
        
        # log-likelihood
        nll = 0.5 * (np.log(2. * np.pi) + logvar_x + torch.exp(-logvar_x) * (x - mu_x)**2).sum()
        
        # final ELBO
        elbo = (nll + kl_z1 + kl_z2 + kl_ze) / x.shape[0]

        return elbo, nll / x.shape[0], kl_z1 / x.shape[0], kl_z2 / x.shape[0], kl_ze / x.shape[0], x_recon

    def sample(self, n_samples, n_points, device='cuda'):
        ze = torch.randn(n_samples, self.ze_dim).to(device)
        ze = ze.unsqueeze(1).expand(n_samples, n_points, self.ze_dim).reshape(n_samples * n_points, self.ze_dim)

        z2 = torch.randn(n_samples * n_points,  self.z2_dim).to(device)

        r1 = self.cmlp_z2_r1(z2, ze)
        mu_z1, logvar_z1 = self.cmlp_r1_z1(r1, ze).chunk(2, 1)
        # mu_z1, logvar_z1 = self.cmlp_z2_z1(z2, ze).chunk(2, 1)
        z1 = self.reparametrization(mu_z1, logvar_z1)

        mu_x, logvar_x = self.cmlp_z1_x(torch.cat([z1, r1], dim=1), ze).chunk(2, 1) # TODO check
        x = self.reparametrization(mu_x, logvar_x)
        x = x.reshape(n_samples, n_points, self.x_dim)
        return x

    def lipschitz_loss(self):
        loss = 1.
        for module in self.modules():
            param_dict = module._parameters
            if 'c' in param_dict:
                loss *= param_dict['c']
        return loss * self.lipschitz_loss_weight


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConditionalTopDownVAE(2, 4, h1_dim=3, h2_dim=20, e_dim=30, ze_dim=50, z1_dim=120, z2_dim=260, r1_dim=322).to(device)
    print('Number of trainable parameters:', count_trainable_parameters(model))
    print(model)

    x = torch.randn(8, 2, 20).to(device)
    elbo, *rest = model(x)
    print(elbo.item())

    x_gen = model.sample(10, 100)
    print(x_gen.shape)

    print(model.lipschitz_loss())