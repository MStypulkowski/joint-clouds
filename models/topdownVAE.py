import numpy as np
import torch
import torch.nn as nn
import wandb

from models.backbones import PointNet, MLP
from utils import count_trainable_parameters


class TopDownVAE(nn.Module):
    def __init__(self, x_dim, y_dim, h1_dim=64, h2_dim=64, e_dim=64, z1_dim=64, z2_dim=64, hid_dim=64, n_layers=3, activation='relu'):
        super(TopDownVAE, self).__init__()

        assert h2_dim == z1_dim
        self.z2_dim = z2_dim

        # default option for PointNet model is not to perform pooling and fully conected layers
        self.pn_xh1 = PointNet(x_dim, h1_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation)
        self.pn_z2h2 = PointNet(z2_dim, h2_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation)
        self.pn_h2z1 = PointNet(h2_dim, 2 * z1_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation)
        self.pn_h1z1 = PointNet(h1_dim, 2 * z1_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation)
        self.pn_z1x = PointNet(z1_dim, 2 * x_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation)

        self.mlp_he = MLP(h1_dim, e_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation)
        # self.mlp_ey = MLP(e_dim, y_dim, n_layers=n_layers)
        self.mlp_ez2 = MLP(e_dim, 2 * z2_dim, hid_dim=hid_dim, n_layers=n_layers, activation=activation)
    
    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, only_classify=False, epoch=None):
        # bottom-up deterministic path
        # print('x', x.shape)
        # x -> N, 2, M
        h = self.pn_xh1(x) # h -> N, h_dim, M
        # print('h', h.shape)
        e = torch.max(h, 2, keepdim=True)[0]
        e = e.squeeze(-1) # pre e -> N, h_dim
        # print('h_pooled', e.shape)
        e = self.mlp_he(e) # e -> N, e_dim
        # e_log = e.detach().cpu().numpy()
        # e_log = [[e1, e2] for (e1, e2) in e_log]
        # table = wandb.Table(data=e_log, columns=['e1', 'e2'])
        # wandb.log({f'e epoch {epoch}': wandb.plot.scatter(table, 'e1', 'e2', title=f'e {epoch}')})
        # print('e', e.shape)

        #classification
        # logits = self.mlp_ey(e) # N, n_classes
        # print('logits', logits.shape)

        # if only_classify:
        #     return logits

        # top-down stochastic path
        delta_mu2, delta_logvar2 = self.mlp_ez2(e).chunk(2, 1) # delta_mu2 = delta_logvar2 -> N, z2_dim
        # if epoch is not None:
        #     wandb.log({'delta_mu2 mean': delta_mu2.mean(),
        #             'delta_mu2 std': delta_mu2.std(),
        #             'delta_logvar2 mean': delta_logvar2.mean(),
        #             'delta_logvar2 std': delta_logvar2.std(),
        #     })
        #     _delta_mu2 = delta_mu2.detach().cpu().numpy().reshape(-1)
        #     _delta_mu2 = [[d] for d in _delta_mu2]
        #     _delta_mu2 = wandb.Table(data=_delta_mu2, columns=['data'])
        #     wandb.log({f'_delta_mu2 {epoch}': wandb.plot.histogram(_delta_mu2, 'data', title=f'_delta_mu2 {epoch}')})
        #     _delta_logvar2 = delta_logvar2.detach().cpu().numpy().reshape(-1)
        #     _delta_logvar2 = [[d] for d in _delta_logvar2]
        #     _delta_logvar2 = wandb.Table(data=_delta_logvar2, columns=['data'])
        #     wandb.log({f'_delta_logvar2 {epoch}': wandb.plot.histogram(_delta_logvar2, 'data', title=f'_delta_logvar2 {epoch}')})
            
        # print('delta2_pre', delta_mu2.shape, delta_logvar2.shape)
        delta_mu2, delta_logvar2 = delta_mu2.unsqueeze(-1).expand(-1, self.z2_dim, x.shape[-1]), delta_logvar2.unsqueeze(-1).expand(-1, self.z2_dim, x.shape[-1]) # delta_mu2 = delta_logvar2 -> N, z2_dim, M
        # print('delta2', delta_mu2.shape, delta_logvar2.shape)
        z2 = self.reparametrization(delta_mu2, delta_logvar2) # N, z2_dim, M
        # print('z2', z2.shape)

        delta_mu1, delta_logvar1 = self.pn_h1z1(h).chunk(2, 1) # delta_mu1 = delta_logvar1 -> N, z1_dim, M
        # if epoch is not None:
        #     wandb.log({'delta_mu1 mean': delta_mu1.mean(),
        #             'delta_mu1 std': delta_mu1.std(),
        #             'delta_logvar1 mean': delta_logvar1.mean(),
        #             'delta_logvar1 std': delta_logvar1.std(),
        #     })
        #     _delta_mu1 = delta_mu1.detach().cpu().numpy().reshape(-1)
        #     _delta_mu1 = [[d] for d in _delta_mu1]
        #     _delta_mu1 = wandb.Table(data=_delta_mu1, columns=['data'])
        #     wandb.log({f'_delta_mu1 {epoch}': wandb.plot.histogram(_delta_mu1, 'data', title=f'_delta_mu1 {epoch}')})
        #     _delta_logvar1 = delta_logvar1.detach().cpu().numpy().reshape(-1)
        #     _delta_logvar1 = [[d] for d in _delta_logvar1]
        #     _delta_logvar1 = wandb.Table(data=_delta_logvar1, columns=['data'])
        #     wandb.log({f'_delta_logvar1 {epoch}': wandb.plot.histogram(_delta_logvar1, 'data', title=f'_delta_logvar1 {epoch}')})

        # print('delta1', delta_mu1.shape, delta_logvar1.shape)
        h2 = self.pn_z2h2(z2)
        mu1, logvar1 = self.pn_h2z1(h2).chunk(2, 1) # mu1 = logvar1 -> N, z1_dim, M
        # if epoch is not None:
        #     wandb.log({'mu1 mean': mu1.mean(),
        #             'mu1 std': mu1.std(),
        #             'logvar1 mean': logvar1.mean(),
        #             'logvar1 std': logvar1.std(),
        #     })
        #     _mu1 = mu1.detach().cpu().numpy().reshape(-1)
        #     _mu1 = [[d] for d in _mu1]
        #     table_mu = wandb.Table(data=_mu1, columns=['data'])
        #     wandb.log({f'_mu1 {epoch}': wandb.plot.histogram(table_mu, 'data', title=f'_mu1 {epoch}')})
        #     _logvar1 = logvar1.detach().cpu().numpy().reshape(-1)
        #     _logvar1 = [[d] for d in _logvar1]
        #     _logvar1 = wandb.Table(data=_logvar1, columns=['data'])
        #     wandb.log({f'_logvar1 {epoch}': wandb.plot.histogram(_logvar1, 'data', title=f'_logvar1 {epoch}')})

        # print('params1_pre', mu1.shape, logvar1.shape)
        # mu1, logvar1 = mu1.unsqueeze(-1).expand(delta_mu1.shape), logvar1.unsqueeze(-1).expand(delta_logvar1.shape)
        # print('params1', mu1.shape, logvar1.shape)
        z1 = self.reparametrization(mu1 + delta_mu1, logvar1 + delta_logvar1) # z1 -> N, z1_dim, M
        z1 += h2
        # print('z1', z1.shape)
        mu_x, logvar_x = self.pn_z1x(z1).chunk(2, 1) # mu_x, logvar_x -> N, 2, M
        # if epoch is not None:
        #     wandb.log({'mu_x mean': mu_x.mean(),
        #             'mu_x std': mu_x.std(),
        #             'logvar_x mean': logvar_x.mean(),
        #             'logvar_x std': logvar_x.std(),
        #     })
        #     _mu_x = mu_x.detach().cpu().numpy().reshape(-1)
        #     _mu_x = [[d] for d in _mu_x]
        #     _mu_x = wandb.Table(data=_mu_x, columns=['data'])
        #     wandb.log({f'_mu_x {epoch}': wandb.plot.histogram(table_mu, 'data', title=f'_mu_x {epoch}')})
        #     _logvar_x = logvar_x.detach().cpu().numpy().reshape(-1)
        #     _logvar_x = [[d] for d in _logvar_x]
        #     _logvar_x = wandb.Table(data=_logvar_x, columns=['data'])
        #     wandb.log({f'_logvar_x {epoch}': wandb.plot.histogram(_logvar_x, 'data', title=f'_logvar_x {epoch}')})
        # print('params_x', mu_x.shape, logvar_x.shape)

        # ELBO
        # KL
        kl_z2 = 0.5 * (delta_mu2**2 + torch.exp(delta_logvar2) - delta_logvar2 - 1).sum([1, 2])
        # print('kl_z2', kl_z2.shape)
        kl_z1 = 0.5 * (delta_mu1**2 * torch.exp(-logvar1) + torch.exp(delta_logvar1) - delta_logvar1 - 1).sum([1, 2])
        # print('kl_z1', kl_z1.shape)
        
        # log-likelihood
        nll = 0.5 * (np.log(2. * np.pi) + logvar_x + torch.exp(-logvar_x) * (x - mu_x)**2).sum([1, 2])
        # print('nll', nll.shape)
        
        # final ELBO
        elbo = (nll + kl_z2 + kl_z1).mean()

        # return elbo, logits, nll.mean(), kl_z1.mean(), kl_z2.mean()
        return elbo, nll.mean(), kl_z1.mean(), kl_z2.mean()

    def sample(self, n_samples, n_points, device='cuda'):
        z2 = torch.randn(n_samples, self.z2_dim, n_points).to(device)
        # print('z2', z2.shape)
        h2 = self.pn_z2h2(z2)
        mu1, logvar1 = self.pn_h2z1(h2).chunk(2, 1)
        # print('params1', mu1.shape, logvar1.shape)
        z1 = self.reparametrization(mu1, logvar1)
        z1 += h2
        # print('z1', z1.shape)
        mu_x, logvar_x = self.pn_z1x(z1).chunk(2, 1)
        # print('params_x', mu_x.shape, logvar_x.shape)
        x = self.reparametrization(mu_x, logvar_x)

        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hvae = TopDownVAE(2, 4, h1_dim=32, h2_dim=34, e_dim=64, z1_dim=128, z2_dim=256).to(device)
    print('Number of trainable parameters:', count_trainable_parameters(hvae))

    x = torch.randn(8, 2, 16).to(device)
    loss, logits = hvae(x)
    print(loss.item(), logits.shape)

    x_gen = hvae.sample(10, 100)
    print(x_gen.shape)