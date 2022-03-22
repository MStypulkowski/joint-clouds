import torch
import torch.nn as nn

from models.backbones import MLP, CMLP
from utils import reparametrization


class H_Block(nn.Module):
    def __init__(self, h_prev_dim, h_dim, z_dim, ze_dim,
                hid_dim=64, n_resnet_blocks=1, activation='relu', last_activation=None, 
                use_batchnorms=False, use_lipschitz_norm=False):
        super(H_Block, self).__init__()

        self.mlp_h = MLP(h_prev_dim, h_dim, hid_dim=hid_dim, n_resnet_blocks=n_resnet_blocks, 
                        activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                        use_lipschitz_norm=use_lipschitz_norm)
        self.cmlp_z = CMLP(h_dim, 2 * z_dim, ze_dim, hid_dim=hid_dim, n_resnet_blocks=n_resnet_blocks, 
                        activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                        use_lipschitz_norm=use_lipschitz_norm)

        self.h = None
        self.delta_mu_z, self.delta_logvar_z = None, None

    def forward(self, h_prev):
        self.h = self.mlp_h(h_prev)
        return self.h
    
    def calculate_deltas(self, ze):
        self.delta_mu_z, self.delta_logvar_z = self.cmlp_z(self.h, ze).chunk(2, 1)

    def get_params(self):
        return self.delta_mu_z, self.delta_logvar_z


class Z_Block(nn.Module):
    def __init__(self, z_dim, r_dim, z_next_dim, ze_dim, 
                hid_dim=64, n_resnet_blocks=1, activation='relu', last_activation=None, 
                use_batchnorms=False, use_lipschitz_norm=False):
        super(Z_Block, self).__init__()

        self.cmlp_z_prev_r= CMLP(z_dim, r_dim, ze_dim, hid_dim=hid_dim, n_resnet_blocks=n_resnet_blocks, 
                            activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                            use_lipschitz_norm=use_lipschitz_norm)
        self.cmlp_r_z= CMLP(r_dim, 2 * z_next_dim, ze_dim, hid_dim=hid_dim, n_resnet_blocks=n_resnet_blocks, 
                            activation=activation, last_activation=last_activation, use_batchnorms=use_batchnorms, 
                            use_lipschitz_norm=use_lipschitz_norm)
        
        self.mu_z, self.logvar_z = None, None

    def forward(self, z_prev, ze, delta_mu_z, delta_logvar_z):
        r = self.cmlp_z_prev_r(z_prev, ze)
        mu_z, logvar_z = self.cmlp_r_z(r, ze).chunk(2, 1)
        self.mu_z, self.logvar_z = mu_z, logvar_z
        z = reparametrization(mu_z + delta_mu_z, logvar_z + delta_logvar_z)
        return torch.cat([z, r], dim=1)

    def get_params(self):
        return self.mu_z, self.logvar_z