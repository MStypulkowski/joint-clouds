import numpy as np
import torch
import torch.nn as nn

from models.pointnet import PointNet, MLP


class HierarchicalVAE(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim=64, e_dim=64, z1_dim=64, z2_dim=64, n_layers=3):
        super(HierarchicalVAE, self).__init__()

        self.z2_dim = z2_dim

        # default option for PointNet model is not to perform pooling and fully conected layers
        self.pn_xh = PointNet(x_dim, h_dim, n_layers=n_layers)
        self.pn_z2z1 = PointNet(z2_dim, 2 * z1_dim, n_layers=n_layers)
        self.pn_hz1 = PointNet(h_dim, 2 * z1_dim, n_layers=n_layers)
        self.pn_z1x = PointNet(z1_dim, 2 * x_dim, n_layers=n_layers)

        self.mlp_he = MLP(h_dim, e_dim, n_layers=n_layers)
        self.mlp_ey = MLP(e_dim, y_dim, n_layers=n_layers)
        self.mlp_ez2 = MLP(e_dim, 2 * z2_dim, n_layers=n_layers)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, only_classify=False):
        # bottom-up deterministic path
        # print('x', x.shape)
        h = self.pn_xh(x)
        # print('h', h.shape)
        e = torch.max(h, 2, keepdim=True)[0]
        e = e.squeeze(-1)
        # print('h_pooled', e.shape)
        e = self.mlp_he(e)
        # print('e', e.shape)

        #classification
        logits = self.mlp_ey(e)
        # print('logits', logits.shape)

        if only_classify:
            return logits

        # top-down stochastic path
        delta_mu2, delta_logvar2 = self.mlp_ez2(e).chunk(2, 1)
        # print('delta2_pre', delta_mu2.shape, delta_logvar2.shape)
        delta_mu2, delta_logvar2 = delta_mu2.unsqueeze(-1).expand(-1, self.z2_dim, x.shape[-1]), delta_logvar2.unsqueeze(-1).expand(-1, self.z2_dim, x.shape[-1])
        # print('delta2', delta_mu2.shape, delta_logvar2.shape)
        z2 = self.reparametrization(delta_mu2, delta_logvar2)
        # print('z2', z2.shape)

        delta_mu1, delta_logvar1 = self.pn_hz1(h).chunk(2, 1)
        # print('delta1', delta_mu1.shape, delta_logvar1.shape)
        mu1, logvar1 = self.pn_z2z1(z2).chunk(2, 1)
        # print('params1_pre', mu1.shape, logvar1.shape)
        # mu1, logvar1 = mu1.unsqueeze(-1).expand(delta_mu1.shape), logvar1.unsqueeze(-1).expand(delta_logvar1.shape)
        # print('params1', mu1.shape, logvar1.shape)
        z1 = self.reparametrization(mu1 + delta_mu1, logvar1 + delta_logvar1)
        # print('z1', z1.shape)

        mu_x, logvar_x = self.pn_z1x(z1).chunk(2, 1)
        # print('params_x', mu_x.shape, logvar_x.shape)

        # ELBO
        # KL
        kl_z2 = 0.5 * (delta_mu2**2 + torch.exp(delta_logvar2) - delta_logvar2 - 1).sum([1, 2])
        # print('kl_z2', kl_z2.shape)
        kl_z1 = 0.5 * (delta_mu1**2 / torch.exp(logvar1) + torch.exp(delta_logvar1) - delta_logvar1 - 1).sum([1, 2])
        # print('kl_z1', kl_z1.shape)

        # log-likelihood
        nll = 0.5 * (np.log(2. * np.pi) + logvar_x + torch.exp(-logvar_x) * (x - mu_x)**2).sum([1, 2])
        # print('nll', nll.shape)

        # final ELBO
        elbo = (nll + kl_z2 + kl_z1).mean()

        return elbo, logits

    def sample(self, n_samples, n_points, device='cuda'):
        z2 = torch.randn(n_samples, self.z2_dim, n_points).to(device)
        # print('z2', z2.shape)
        mu1, logvar1 = self.pn_z2z1(z2).chunk(2, 1)
        # print('params1', mu1.shape, logvar1.shape)
        z1 = self.reparametrization(mu1, logvar1)
        # print('z1', z1.shape)
        mu_x, logvar_x = self.pn_z1x(z1).chunk(2, 1)
        # print('params_x', mu_x.shape, logvar_x.shape)
        x = self.reparametrization(mu_x, logvar_x)

        return x