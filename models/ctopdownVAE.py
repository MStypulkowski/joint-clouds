import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from models.backbones import MLP, CMLP, PositionalEncoding#, PointNet, CPointNet
from models.scale_blocks import H_Block, Z_Block
from utils import count_trainable_parameters, reparametrization, get_kl


class ConditionalTopDownVAE(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim=64, e_dim=64, ze_dim=64, z_dim=64,
                n_latents=2, encoder_hid_dim=64, decoder_hid_dim=64, encoder_n_resnet_blocks=1, decoder_n_resnet_blocks=1,
                activation='relu', last_activation=None, use_batchnorms=False, use_lipschitz_norm=False, lipschitz_loss_weight=1e-6, 
                use_positional_encoding=False, L=2):
        super(ConditionalTopDownVAE, self).__init__()

        self.ze_dim = ze_dim
        self.z_dim = z_dim
        self.x_dim= x_dim
        self.h_dim = h_dim

        self.n_latent = n_latents

        self.use_lipschitz_norm = use_lipschitz_norm
        self.lipschitz_loss_weight = lipschitz_loss_weight

        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(L, x_dim)


        self.h_blocks = []
        self.z_blocks = []
        for i in range(n_latents):
            if i == 0:
                if use_positional_encoding:
                    h_in_dim = 2 * L * x_dim
                else:
                    h_in_dim = x_dim
            else:
                h_in_dim = h_dim
            self.h_blocks.append(
                H_Block(h_in_dim, h_dim, z_dim, ze_dim, i, n_latents, hid_dim=encoder_hid_dim, n_resnet_blocks=encoder_n_resnet_blocks, 
                        activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                        use_lipschitz_norm=use_lipschitz_norm)
            )

            if i > 0:
                if i == 1:
                    z_in_dim = z_dim
                else:
                    z_in_dim = 2 * z_dim # because of concatenation of z and r from previous z_block
                self.z_blocks.append(
                    Z_Block(z_in_dim, z_dim, z_dim, ze_dim, hid_dim=decoder_hid_dim, n_resnet_blocks=decoder_n_resnet_blocks, 
                            activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                            use_lipschitz_norm=use_lipschitz_norm)
                )
                
        self.h_blocks = nn.ModuleList(self.h_blocks)
        self.z_blocks = nn.ModuleList(self.z_blocks)

        self.cmlp_z_x= CMLP(z_dim + z_dim, 2 * x_dim, ze_dim, hid_dim=decoder_hid_dim, n_resnet_blocks=decoder_n_resnet_blocks, 
                            activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                            use_lipschitz_norm=use_lipschitz_norm)

        self.mlp_h_e = MLP(h_dim, e_dim, hid_dim=encoder_hid_dim, n_resnet_blocks=encoder_n_resnet_blocks, 
                            activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                            use_lipschitz_norm=use_lipschitz_norm)
                            
        self.mlp_e_ze = MLP(e_dim, 2 * ze_dim, hid_dim=encoder_hid_dim, n_resnet_blocks=encoder_n_resnet_blocks, 
                            activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                            use_lipschitz_norm=use_lipschitz_norm)
        
    def forward(self, x, only_classify=False, epoch=None, save_dir=None):
        # bottom-up deterministic path
        # x - N, M, x_dim
        x_encoded = x + 0.
        h = x.reshape(-1, self.x_dim) # N*M, x_dim

        if self.use_positional_encoding:
            h = self.positional_encoding.encode(h)

        for h_block in self.h_blocks:
            h = h_block(h)

        h = h.reshape(-1, x.shape[1], self.h_dim) # N, M, h2_dim
        e = torch.max(h, 1, keepdim=True)[0] # N, 1, h2_dim
        e = e.squeeze(1) # N, h2_dim
        e = self.mlp_h_e(e) # N, e_dim
        delta_mu_ze, delta_logvar_ze = self.mlp_e_ze(e).chunk(2, 1) # N, ze_dim
        # delta_logvar_ze = F.hardtanh(delta_logvar_ze)

        # top-down stochastic path
        ze = reparametrization(delta_mu_ze, delta_logvar_ze) # N x ze_dim
        ze = ze.unsqueeze(1).expand(-1, x.shape[1], self.ze_dim).reshape(-1, self.ze_dim) # N*M, ze_dim but in correct order
        
        self.h_blocks[-1].calculate_deltas(ze)
        z = reparametrization(self.h_blocks[-1].delta_mu_z, self.h_blocks[-1].delta_logvar_z) # N*M, z2_dim
        if epoch is not None:
            zs_recon = [z.reshape(-1, x.shape[1], self.z_dim)]
        for i in range(self.n_latent - 1):
            self.h_blocks[- i - 2].calculate_deltas(ze, z_prev=z)
            delta_mu_z, delta_logvar_z = self.h_blocks[- i - 2].get_params()
            # delta_logvar_z = F.hardtanh(delta_logvar_z)
            z = self.z_blocks[i](z, ze, delta_mu_z, delta_logvar_z)
            if epoch is not None:
                zs_recon.append(z[:, :self.z_dim].reshape(-1, x.shape[1], self.z_dim))
        
        mu_x, logvar_x = self.cmlp_z_x(z, ze).chunk(2, 1) # N*M, x_dim
        # logvar_x = F.hardtanh(logvar_x)

        # if not self.use_positional_encoding:
        mu_x = mu_x.reshape(-1, x.shape[1], self.x_dim)#.permute(0, 2, 1) # N, M, x_dim
        logvar_x = logvar_x.reshape(-1, x.shape[1], self.x_dim)#.permute(0, 2, 1) # N, M, x_dim

        if epoch is not None:
            x_recon = reparametrization(mu_x, logvar_x)
            # if self.use_positional_encoding:
            #     x_recon = F.hardtanh(x_recon)
            #     x_recon = self.positional_encoding.decode(x_recon).reshape(x.shape)
        
        # ELBO
        # KLs
        kl_ze = get_kl(delta_mu_ze, delta_logvar_ze, torch.zeros_like(delta_mu_ze), torch.zeros_like(delta_logvar_ze)) / x.shape[0]

        kls = OrderedDict({})
        for i in range(self.n_latent):
            delta_mu_z, delta_logvar_z = self.h_blocks[- i - 1].get_params()
            # delta_logvar_z = F.hardtanh(delta_logvar_z)
            if i == 0:
                mu_z, logvar_z = torch.zeros_like(delta_mu_z), torch.zeros_like(delta_logvar_z)
            else:
                mu_z, logvar_z = self.z_blocks[i - 1].get_params()
            # logvar_z = F.hardtanh(logvar_z)
            
            kls['KL_z' + str(self.n_latent - i)] = get_kl(delta_mu_z, delta_logvar_z, mu_z, logvar_z) / x.shape[0]
        
        # NLL
        nll = 0.5 * (np.log(2. * np.pi) + logvar_x + torch.exp(-logvar_x) * (x_encoded - mu_x)**2).sum() / x.shape[0]
        
        # final ELBO
        elbo = nll + kl_ze
        for i in kls:
            elbo += kls[i]

        if epoch is None:
            return elbo, nll, kl_ze, kls
        return elbo, nll, kl_ze, kls, x_recon, zs_recon

    def sample(self, n_samples, n_points, device='cuda'):
        ze = torch.randn(n_samples, self.ze_dim).to(device)
        ze = ze.unsqueeze(1).expand(n_samples, n_points, self.ze_dim).reshape(n_samples * n_points, self.ze_dim)

        z = torch.randn(n_samples * n_points,  self.z_dim).to(device)
        zs = [z.cpu().numpy().reshape(n_samples, n_points, self.z_dim)]
        for i in range(self.n_latent - 1):
            z = self.z_blocks[i](z, ze, 0., 0.)
            zs.append(z[:, :self.z_dim].cpu().numpy().reshape(n_samples, n_points, self.z_dim))
        zs = torch.tensor(zs)

        mu_x, logvar_x = self.cmlp_z_x(z, ze).chunk(2, 1)
        x = reparametrization(mu_x, logvar_x)
        # if self.use_positional_encoding:
        #     x = F.hardtanh(x)
        #     x = self.positional_encoding.decode(x)
        x = x.reshape(n_samples, n_points, self.x_dim)
        return x, zs

    def lipschitz_loss(self):
        assert self.use_lipschitz_norm

        loss = self.lipschitz_loss_weight
        for module in self.modules():
            param_dict = module._parameters
            if 'c' in param_dict:
                loss *= param_dict['c']
        return loss


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConditionalTopDownVAE(2, 4, h_dim=3, e_dim=30, ze_dim=50, z_dim=121, n_latents=3).to(device)
    print('Number of trainable parameters:', count_trainable_parameters(model))
    print(model)

    x = torch.randn(8, 20, 2).to(device)
    elbo, *rest = model(x)
    print(elbo.item(), rest[:-1])

    x_gen = model.sample(10, 100)
    print(x_gen.shape)

    # print(model.lipschitz_loss())