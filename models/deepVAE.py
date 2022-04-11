import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from models.backbones import MLP, CMLP#, PointNet, CPointNet
from models.scale_blocks import H_Block, Z_Block
from utils import count_trainable_parameters, reparametrization, get_kl


def analytical_kl(mu1, mu2, logvar1, logvar2):
    return -0.5 + logvar2 - logvar1 + 0.5 * (logvar1.exp() ** 2 + (mu1 - mu2) ** 2) / (logvar2.exp() ** 2)


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

        self.residual = residual

        self.fc1 = CLinear(in_dim, e_dim, hid_dim)
        self.fc2 = CLinear(hid_dim, e_dim, hid_dim)
        self.fc3 = CLinear(hid_dim, e_dim, out_dim)

        self.activation = nn.SiLU()

    def forward(self, x, e):
        _x = self.activation(self.fc1(x, e))
        _x = self.activation(self.fc2(_x, e))
        _x = self.activation(self.fc3(_x, e))
        # TODO out activation
        return _x + x if self.residual else _x


class TopDownBlock(nn.Module):
    def __init__(self):
        super(TopDownBlock, self).__init__()
        self.q_block = Block(in_dim, e_dim, hid_dim, out_dim, residual=False)
        self.p_block = Block(in_dim, e_dim, hid_dim, out_dim, residual=False)

        self.out_block = Block(in_dim, e_dim, hid_dim, out_dim, residual=True)

        self.fc_z = CLinear(in_dim, e_dim, hid_dim)
        # TODO activation

    def forward(self, z, h, e):
        zh = torch.cat([z, h], dim=1)
        q_mu, q_logvar = self.q_block(zh, e).chunk(2, 1)

        p_out = self.p_block(z, e)
        z_residual, (p_mu, p_logvar) = p_out[:, :z.shape[-1]], p_out[:, z.shape[-1]:].chunk(2, 1)

        z_sample = reparametrization(q_mu, q_logvar)
        z_sample = self.fc_z(z_sample)

        z = z + z_residual + z_sample
        z = self.out_block(z, e)

        kl = analytical_kl(q_mu, p_mu, q_logvar, p_logvar)

        return z, kl
    
    def sample(self, z, e):
        p_out = self.p_block(z, e)
        z_residual, (p_mu, p_logvar) = p_out[:, :z.shape[-1]], p_out[:, z.shape[-1]:].chunk(2, 1)

        z_sample = reparametrization(p_mu, p_logvar)
        z_sample = self.fc_z(z_sample)

        z = z + z_residual + z_sample
        z = self.out_block(z, e)

        return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        pass


class DeepVAE(nn.Module):
    def __init__(self):
        super(DeepVAE, self).__init__()

    def forward(self, x):
        pass