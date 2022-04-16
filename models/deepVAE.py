import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt


def reparametrization(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def analytical_kl(mu1, mu2, logvar1, logvar2):
    return -0.5 + logvar2 - logvar1 + 0.5 * (logvar1.exp() ** 2 + (mu1 - mu2) ** 2) / (logvar2.exp() ** 2)


def gaussian_nll(x, x_mu, x_logvar):
    return 0.5 * (np.log(2. * np.pi) + x_logvar + torch.exp(-x_logvar) * (x - x_mu)**2)


class CLinear(nn.Module):
    def __init__(self, in_dim, e_dim, hid_dim):
        super(CLinear, self).__init__()
        self.fc = nn.Linear(in_dim + e_dim, hid_dim)
    def forward(self, x, e):
        xe = torch.cat([x, e], dim=1) # TODO preprocessing
        return self.fc(xe)


class Block(nn.Module):
    def __init__(self, in_dim, e_dim, hid_dim, out_dim, residual=True):
        super(Block, self).__init__()

        if residual:
            assert in_dim == out_dim

        self.residual = residual

        if e_dim is None:
            self.fc1 = nn.Linear(in_dim, hid_dim)
            self.fc2 = nn.Linear(hid_dim, hid_dim)
            self.fc3 = nn.Linear(hid_dim, out_dim)    
        else:
            self.fc1 = CLinear(in_dim, e_dim, hid_dim)
            self.fc2 = CLinear(hid_dim, e_dim, hid_dim)
            self.fc3 = CLinear(hid_dim, e_dim, out_dim)    

        self.activation = nn.SiLU()

    def forward(self, x, e=None):
        if e is None:
            _x = self.activation(self.fc1(x))
            _x = self.activation(self.fc2(_x))
            _x = self.activation(self.fc3(_x))
        else:
            _x = self.activation(self.fc1(x, e))
            _x = self.activation(self.fc2(_x, e))
            _x = self.activation(self.fc3(_x, e))
        # TODO out activation
        return _x + x if self.residual else _x


class TopDownBlock(nn.Module):
    def __init__(self, z_dim, h_dim, e_dim, hid_dim):
        super(TopDownBlock, self).__init__()

        self.q_block = Block(z_dim + h_dim, e_dim, hid_dim, 2 * z_dim, residual=False)
        self.p_block = Block(z_dim, e_dim, hid_dim, 2 * z_dim, residual=False)

        self.out_block = Block(z_dim, e_dim, hid_dim, z_dim, residual=True)

        self.fc_z = CLinear(z_dim, e_dim, z_dim)
        # TODO activation

    def forward(self, z, h, e):
        zh = torch.cat([z, h], dim=1)
        q_mu, q_logvar = self.q_block(zh, e).chunk(2, 1)

        p_out = self.p_block(z, e)
        z_residual, (p_mu, p_logvar) = p_out[:, :z.shape[-1]], p_out[:, z.shape[-1]:].chunk(2, 1)

        z_sample = reparametrization(q_mu, q_logvar)
        z_sample = self.fc_z(z_sample, e)

        z = z + z_residual + z_sample
        z = self.out_block(z, e)

        kl = analytical_kl(q_mu, p_mu, q_logvar, p_logvar)

        return z, kl
    
    def sample(self, z, e):
        p_out = self.p_block(z, e)
        z_residual, (p_mu, p_logvar) = p_out[:, :z.shape[-1]], p_out[:, z.shape[-1]:].chunk(2, 1)

        z_sample = reparametrization(p_mu, p_logvar)
        z_sample = self.fc_z(z_sample, e)

        z = z + z_residual + z_sample
        z = self.out_block(z, e)

        return z


class PriorBlock(nn.Module):
    # Block for z_e and z_n
    def __init__(self, z_dim, h_dim, e_dim, hid_dim):
        super(PriorBlock, self).__init__()

        self.q_block = Block(h_dim, e_dim, hid_dim, 2 * z_dim, residual=False)

        self.out_block = Block(z_dim, e_dim, hid_dim, z_dim, residual=True)

        if e_dim is None:
            self.fc_z = nn.Linear(z_dim, z_dim)
        else:
            self.fc_z = CLinear(z_dim, e_dim, z_dim)

    def forward(self, h, e=None):
        q_mu, q_logvar = self.q_block(h, e).chunk(2, 1)
        print(q_mu.shape, q_logvar.shape)
        z_sample = reparametrization(q_mu, q_logvar)

        if e is None:
            z_sample = self.fc_z(z_sample)
        else:
            z_sample = self.fc_z(z_sample, e)

        z += z_sample
        z = self.out_block(z, e)

        kl = analytical_kl(q_mu, torch.zeros_like(q_mu), q_logvar, torch.zeros_like(q_logvar))
        print(kl.shape)

        return z, kl

    def sample(self, e=None):
        z_sample = torch.randn()

        if e is None:
            z_sample = self.fc_z(z_sample)
        else:
            z_sample = self.fc_z(z_sample, e)

        z += z_sample
        z = self.out_block(z, e)

        return z


class Encoder(nn.Module):
    def __init__(self, n_latents, x_dim, h_dim, hid_dim, e_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        self.in_block = Block(x_dim, None, hid_dim, h_dim, residual=False)
        self.h_blocks = nn.ModuleList([Block(h_dim, None, hid_dim, h_dim, residual=True) for _ in range(n_latents)])
        self.e_block = Block(h_dim, None, hid_dim, e_dim, residual=True)

    def forward(self, x):
        h = x.reshape(-1, self.x_dim)
        h = self.in_block(h)

        hs = []
        for h_block in self.h_blocks:
            h = h_block(h)
            hs.append(h)
        
        h = h.reshape(-1, x.shape[1], self.h_dim) # N, M, h2_dim
        e = torch.max(h, 1, keepdim=True)[0] # N, 1, h2_dim
        e = e.squeeze(1) # N, h2_dim
        e = self.e_block(h)
        # TODO check if condition hs on e

        return hs, e


class Decoder(nn.Module):
    def __init__(self, n_latents, ze_dim, e_dim, z_dim, h_dim, hid_dim, x_dim, n_points_per_cloud):
        super(Decoder, self).__init__()

        self.ze_dim = ze_dim
        self.n_points_per_cloud = n_points_per_cloud

        self.ze_block = PriorBlock(ze_dim, e_dim, None, hid_dim)
        self.zn_block = PriorBlock(z_dim, h_dim, e_dim, hid_dim)
        self.z_blocks = nn.ModuleList([TopDownBlock(z_dim, h_dim, e_dim, hid_dim) for _ in range(n_latents - 1)])
        self.x_block = Block(z_dim, ze_dim, hid_dim, x_dim, residual=True)

    def forward(self, hs, e):
        ze = self.ze_block(e)
        ze = ze.unsqueeze(1).expand(-1, self.n_points_per_cloud, self.ze_dim).reshape(-1, self.ze_dim)

        z, kl = self.zn_block(hs[-1], ze)

        kls = [kl]
        for i, z_block in enumerate(self.z_blocks):
            z, kl = z_block(z, hs[-i - 2], ze)
            kls.append(kl)
        
        x_params = self.x_block(z)

        return x_params, kls

    def sample(self):
        pass


class DeepVAE(nn.Module):
    def __init__(self, n_latents, x_dim, h_dim, hid_dim, e_dim, ze_dim, z_dim, n_points_per_cloud):
        super(DeepVAE, self).__init__()
        self.encoder = Encoder(n_latents, x_dim, h_dim, hid_dim, e_dim)
        self.decoder = Decoder(n_latents, ze_dim, e_dim, z_dim, h_dim, hid_dim, x_dim, n_points_per_cloud)

    def forward(self, x):
        hs, e = self.encoder(x)
        x_params, kls = self.decoder(hs, e)
        x_mu, x_logvar = x_params.chunk(2, 1)

        nll = gaussian_nll(x, x_mu, x_logvar)

        loss = nll + kls

        log_dict = None

        return loss, log_dict
    
    def sample(self, n_samples, n_point_per_cloud_gen):
        pass



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(3, 5, 2).to(device)
    model = DeepVAE(3, 2, 8, 16, 8, 8, 2, 5).to(device)
    loss, log_dict = model(x)
    print(loss)
