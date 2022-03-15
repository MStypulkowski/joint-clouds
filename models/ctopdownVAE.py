from cmath import log
import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from models.backbones import MLP, CMLP, PointNet, CPointNet
from utils import count_trainable_parameters


class ConditionalTopDownVAE(nn.Module):
    def __init__(self, x_dim, y_dim, h1_dim=64, h2_dim=64, e_dim=64, ze_dim=64, z1_dim=64, z2_dim=64, hid_dim=64, n_layers=3, activation='relu'):
        super(ConditionalTopDownVAE, self).__init__()

        self.ze_dim = ze_dim
        self.z2_dim = z2_dim
        self.x_dim= x_dim
        self.z1_dim= z1_dim

        # default option for PointNet model is not to perform pooling and fully conected layers
        self.pn_x_h1 = PointNet(x_dim, h1_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        self.pn_h1_h2 = PointNet(h1_dim, h2_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        self.pn_h2_z2 = PointNet(h2_dim, 2 * z2_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        #self.cpn_z2_z1 = CPointNet(z2_dim, 2 * z1_dim, ze_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        self.cmlp_z2_z1= CMLP(z2_dim, z1_dim*2, ze_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        self.pn_h1_z1 = PointNet(h1_dim, 2 * z1_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        #self.pn_z1_x = CPointNet(z1_dim, 2 * x_dim, ze_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        self.cmlp_z1_x= CMLP(z1_dim, x_dim*2, ze_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)

        self.mlp_h2_e = MLP(h2_dim, e_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        self.mlp_e_ze = MLP(e_dim, 2 * ze_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        #self.mlp_ze_z2 = MLP(ze_dim, 2 * z2_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation, last_activation=None)
        
    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, only_classify=False, epoch=None):
        # bottom-up deterministic path
        h1 = self.pn_x_h1(x)
        delta_mu_z1, delta_logvar_z1 = self.pn_h1_z1(h1).chunk(2, 1)
        # print('pre_delta_1', delta_mu_z1.shape, delta_logvar_z1.shape)
        delta_mu_z1, delta_logvar_z1 = delta_mu_z1.permute(0, 2, 1).reshape(-1, self.z1_dim), delta_logvar_z1.permute(0, 2, 1).reshape(-1, self.z1_dim)
        # print('delta_1', delta_mu_z1.shape, delta_logvar_z1.shape)
        # if epoch is not None:
        #     plt.figure()
        #     plt.hist(delta_mu_z1.detach().cpu().numpy().reshape(-1), bins=25)
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/delta_mu_z1.png')
        #     plt.figure()
        #     plt.hist(delta_logvar_z1.detach().cpu().numpy().reshape(-1), bins=25)
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/delta_logvar_z1.png')

        h2 = self.pn_h1_h2(h1)
        delta_mu_z2, delta_logvar_z2 = self.pn_h2_z2(h2).chunk(2, 1)
        # print('pre_delta_2', delta_mu_z2.shape, delta_logvar_z2.shape)
        delta_mu_z2, delta_logvar_z2 = delta_mu_z2.permute(0, 2, 1).reshape(-1, self.z2_dim), delta_logvar_z2.permute(0, 2, 1).reshape(-1, self.z2_dim)
        # print('delta_2', delta_mu_z2.shape, delta_logvar_z2.shape)
        # if epoch is not None:
        #     plt.figure()
        #     plt.hist(delta_mu_z2.detach().cpu().numpy().reshape(-1), bins=25)
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/delta_mu_z2.png')
        #     plt.figure()
        #     plt.hist(delta_logvar_z2.detach().cpu().numpy().reshape(-1), bins=25)
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/delta_logvar_z2.png')

        e = torch.max(h2, 2, keepdim=True)[0]
        e = e.squeeze(-1)
        e = self.mlp_h2_e(e)
        delta_mu_ze, delta_logvar_ze = self.mlp_e_ze(e).chunk(2, 1)
        # if epoch is not None:
        #     plt.figure()
        #     plt.hist(delta_mu_ze.detach().cpu().numpy().reshape(-1), bins=25)
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/delta_mu_ze.png')
        #     plt.figure()
        #     plt.hist(delta_logvar_ze.detach().cpu().numpy().reshape(-1), bins=25)
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/delta_logvar_ze.png')

        # top-down stochastic path
        ze = self.reparametrization(delta_mu_ze, delta_logvar_ze) # N x ze_dim
        ze = ze.unsqueeze(1).expand(-1, x.shape[-1], self.ze_dim).reshape(-1, self.ze_dim) # N * M x ze_dim but in correct order
        # if epoch is not None:
        #     plt.figure()
        #     plt.hist(ze.detach().cpu().numpy().reshape(-1), bins=25)
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/ze.png')
        #mu_z2, logvar_z2 = self.mlp_ze_z2(ze).chunk(2, 1)
        #mu_z2, logvar_z2 = mu_z2.unsqueeze(-1).expand(-1, self.z2_dim, x.shape[-1]), logvar_z2.unsqueeze(-1).expand(-1, self.z2_dim, x.shape[-1])
        
        z2= self.reparametrization(delta_mu_z2, delta_logvar_z2)
        # z2= z2.reshape(-1, self.z2_dim)
        #z2 = self.reparametrization(mu_z2 + delta_mu_z2, logvar_z2 + delta_logvar_z2)
        # if epoch is not None:
        #     _z2 = z2[0].detach().cpu().numpy().T
        #     plt.figure()
        #     plt.scatter(_z2[:, 0], _z2[:, 1])
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/z2.png')
        # print("z2", z2.shape)
        # print("ze", ze.shape)
        mu_z1, logvar_z1 = self.cmlp_z2_z1(z2, ze).chunk(2, 1)
        
        # delta_mu_z1= delta_mu_z1.reshape(-1, self.z1_dim)
        # delta_logvar_z1= delta_logvar_z1.reshape(-1,self.z1_dim)
        # print('pre z1', mu_z1.shape, delta_mu_z1.shape)
        z1 = self.reparametrization(mu_z1 + delta_mu_z1, logvar_z1 + delta_logvar_z1)
        
        #z1 += z2
        # if epoch is not None:
        #     _z1 = z1[0].detach().cpu().numpy().T
        #     plt.figure()
        #     plt.scatter(_z1[:, 0], _z1[:, 1])
        #     plt.savefig('/pio/scratch/1/mstyp/joint-clouds/results/figures/z1.png')
        mu_x, logvar_x = self.cmlp_z1_x(z1, ze).chunk(2, 1)
        mu_x = mu_x.reshape(-1, x.shape[-1], self.x_dim).permute(0, 2, 1)
        logvar_x = logvar_x.reshape(-1, x.shape[-1], self.x_dim).permute(0, 2, 1)
        x_recon = self.reparametrization(mu_x, logvar_x)
        # print(x.shape, mu_x.shape, logvar_x.shape, x_recon.shape)
        
        # ELBO
        # KL
        kl_ze = 0.5 * (delta_mu_ze**2 + torch.exp(delta_logvar_ze) - delta_logvar_ze - 1).sum()
        kl_z2 = 0.5 * (delta_mu_z2**2 + torch.exp(delta_logvar_z2) - delta_logvar_z2 - 1).sum()
        #kl_z2 = 0.5 * (delta_mu_z2**2 * torch.exp(-logvar_z2) + torch.exp(delta_logvar_z2) - delta_logvar_z2 - 1).sum([1, 2])
        kl_z1 = 0.5 * (delta_mu_z1**2 * torch.exp(-logvar_z1) + torch.exp(delta_logvar_z1) - delta_logvar_z1 - 1).sum()
        
        # log-likelihood
        nll = 0.5 * (np.log(2. * np.pi) + logvar_x + torch.exp(-logvar_x) * (x - mu_x)**2).sum()
        
        # final ELBO
        elbo = (nll + kl_z1 + kl_z2 + kl_ze) / x.shape[0]

        # return elbo, logits, nll.mean(), kl_z1.mean(), kl_z2.mean()
        return elbo, nll.mean(), kl_z1.mean(), kl_z2.mean(), kl_ze.mean(), x_recon

    def sample(self, n_samples, n_points, device='cuda'):
        ze = torch.randn(n_samples, self.ze_dim).to(device)
        ze = ze.unsqueeze(1).expand(-1, n_points, self.ze_dim).reshape(-1, self.ze_dim)

        z2 = torch.randn(n_samples * n_points,  self.z2_dim).to(device)
        #mu_z2, logvar_z2 = self.mlp_ze_z2(ze).chunk(2, 1)
        #mu_z2, logvar_z2 = mu_z2.unsqueeze(-1).expand(-1, self.z2_dim, n_points), logvar_z2.unsqueeze(-1).expand(-1, self.z2_dim, n_points)
        #z2 = self.reparametrization(mu_z2, logvar_z2)
        # z2 = torch.randn(n_samples, self.z2_dim, n_points).to(device)
        mu_z1, logvar_z1 = self.cmlp_z2_z1(z2, ze).chunk(2, 1)
        z1 = self.reparametrization(mu_z1, logvar_z1)

        mu_x, logvar_x = self.cmlp_z1_x(z1, ze).chunk(2, 1)
        x = self.reparametrization(mu_x, logvar_x)
        x = x.reshape(-1, n_points, self.x_dim)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConditionalTopDownVAE(2, 4, h1_dim=3, h2_dim=20, e_dim=30, ze_dim=50, z1_dim=120, z2_dim=260).to(device)
    print('Number of trainable parameters:', count_trainable_parameters(model))
    print(model)

    x = torch.randn(8, 2, 20).to(device)
    loss, *rest = model(x)
    print(loss.item())

    x_gen = model.sample(10, 100)
    print(x_gen.shape)

    # x = torch.randint(0, 10, (2, 4))
    # print(x)
    # print(x.unsqueeze(1))