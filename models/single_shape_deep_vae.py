from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import PositionalEncoding
from models.utils import reparametrization, analytical_kl, gaussian_nll


class CLinear(nn.Module):
    def __init__(self, in_dim, c_dim, hid_dim):
        super(CLinear, self).__init__()
        self.fc_x = nn.Linear(in_dim, hid_dim // 2)
        self.fc_c = nn.Linear(c_dim, hid_dim // 2)
        self.fc = nn.Linear(hid_dim, hid_dim)

        self.activation = nn.SiLU()

    def forward(self, x, c):
        x = self.fc_x(x)
        c = self.fc_c(c)
        xc = torch.cat([x, c], dim=1)
        xc = self.activation(xc)
        return self.fc(xc)


class Block(nn.Module):
    def __init__(self, in_dim, c_dim, hid_dim, out_dim, n_layers, residual=True):
        super(Block, self).__init__()

        assert n_layers > 2

        if residual:
            assert in_dim == out_dim

        self.residual = residual

        if c_dim is None:
            self.fc_in = nn.Linear(in_dim, hid_dim)
            self.fcs = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(n_layers - 2)])
            self.fc_out = nn.Linear(hid_dim, out_dim)    
        else:
            self.fc_in = CLinear(in_dim, c_dim, hid_dim)
            self.fcs = nn.ModuleList([CLinear(hid_dim, c_dim, hid_dim) for _ in range(n_layers - 2)])
            self.fc_out = CLinear(hid_dim, c_dim, out_dim)    

        self.activation = nn.SiLU()

    def forward(self, x, c=None):
        if c is None:
            _x = self.activation(self.fc_in(x))
            for fc in self.fcs:
                _x = self.activation(fc(_x))
            _x = self.activation(self.fc_out(_x))
        else:
            _x = self.activation(self.fc_in(x, c))
            for fc in self.fcs:
                _x = self.activation(fc(_x, c))
            _x = self.fc_out(_x, c)
        
        return self.activation(_x + x) if self.residual else _x


class TopDownBlock(nn.Module):
    def __init__(self, z_dim, h_dim, hid_dim, n_layers):
        super(TopDownBlock, self).__init__()

        self.q_block = Block(z_dim + h_dim, None, hid_dim, 2 * z_dim, n_layers, residual=False)
        self.p_block = Block(z_dim, None, hid_dim, 3 * z_dim, n_layers, residual=False)

        self.out_block = Block(z_dim, None, hid_dim, z_dim, n_layers, residual=True)

        self.fc_z = nn.Linear(z_dim, z_dim)
        # TODO activation

    def forward(self, z, h):
        zh = torch.cat([z, h], dim=1)
        q_mu, q_logvar = self.q_block(zh).chunk(2, 1)

        p_out = self.p_block(z)
        z_residual, (p_mu, p_logvar) = p_out[:, :z.shape[-1]], p_out[:, z.shape[-1]:].chunk(2, 1)

        z_sample = reparametrization(q_mu, q_logvar)
        z_sample = self.fc_z(z_sample)

        z = z + z_residual + z_sample
        z = self.out_block(z)

        kl = analytical_kl(q_mu, p_mu, q_logvar, p_logvar)

        assert z.shape == kl.shape

        return z, kl.sum()
    
    def sample(self, z):
        p_out = self.p_block(z)
        z_residual, (p_mu, p_logvar) = p_out[:, :z.shape[-1]], p_out[:, z.shape[-1]:].chunk(2, 1)

        z_sample = reparametrization(p_mu, p_logvar)
        z_sample = self.fc_z(z_sample)

        z = z + z_residual + z_sample
        z = self.out_block(z)

        return z


class PriorBlock(nn.Module):
    # Block for z_e and z_n
    def __init__(self, z_dim, h_dim, hid_dim, n_layers):
        super(PriorBlock, self).__init__()

        self.z_dim = z_dim

        self.q_block = Block(h_dim, None, hid_dim, 2 * z_dim, n_layers, residual=False)

        self.out_block = Block(z_dim, None, hid_dim, z_dim, n_layers, residual=True)

        self.fc_z = nn.Linear(z_dim, z_dim)
        # TODO check if activation

    def forward(self, h):
        q_mu, q_logvar = self.q_block(h).chunk(2, 1)
        z_sample = reparametrization(q_mu, q_logvar)

        z_sample = self.fc_z(z_sample)

        z = self.out_block(z_sample)

        kl = analytical_kl(q_mu, torch.zeros_like(q_mu), q_logvar, torch.zeros_like(q_logvar))

        return z, kl.sum()

    def sample(self, n_samples, device='cuda'):
        z_sample = torch.randn(n_samples, self.z_dim).to(device)

        z_sample = self.fc_z(z_sample)

        z = self.out_block(z_sample)

        return z


class Encoder(nn.Module):
    def __init__(self, n_latents, x_dim, h_dim, hid_dim, n_layers):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        in_dim = x_dim

        self.in_block = Block(in_dim, None, hid_dim, h_dim, n_layers, residual=False)
        self.h_blocks = nn.ModuleList([Block(h_dim, None, hid_dim, h_dim, n_layers, residual=True) for _ in range(n_latents)])

    def forward(self, x):
        h = x.reshape(-1, self.x_dim)

        h = self.in_block(h)

        hs = []
        for h_block in self.h_blocks:
            h = h_block(h)
            hs.append(h)
        
        h = h.reshape(-1, x.shape[1], self.h_dim) # N, M, h2_dim

        return hs


class Decoder(nn.Module):
    def __init__(self, n_latents, z_dim, h_dim, hid_dim, x_dim, n_layers):
        super(Decoder, self).__init__()

        self.zn_block = PriorBlock(z_dim, h_dim, hid_dim, n_layers)
        self.z_blocks = nn.ModuleList([TopDownBlock(z_dim, h_dim, hid_dim, n_layers) for _ in range(n_latents - 1)])

        out_dim = x_dim

        self.x_block = Block(z_dim, None, hid_dim, 2 * out_dim, n_layers, residual=False)

    def forward(self, hs):
        z, kl_zn = self.zn_block(hs[-1])

        kls = [kl_zn]
        for i, z_block in enumerate(self.z_blocks):
            z, kl = z_block(z, hs[-i - 1])
            kls.append(kl)
        
        x_mu, x_logvar = self.x_block(z).chunk(2, 1)
        x_recon = reparametrization(x_mu, x_logvar)

        return x_mu, x_logvar, kls, x_recon

    def sample(self, n_samples, n_points_per_cloud_gen, device='cuda'):
        z = self.zn_block.sample(n_samples * n_points_per_cloud_gen, device=device)
        for z_block in self.z_blocks:
            z = z_block.sample(z)

        x_mu, x_logvar = self.x_block(z).chunk(2, 1)
        
        x_sample = reparametrization(x_mu, x_logvar)

        return x_sample.reshape(n_samples, n_points_per_cloud_gen, -1)


class SingleShapeDeepVAE(nn.Module):
    def __init__(self, n_latents, x_dim, h_dim, hid_dim, z_dim, n_layers):
        super(SingleShapeDeepVAE, self).__init__()
        self.x_dim = x_dim

        self.encoder = Encoder(n_latents, x_dim, h_dim, hid_dim, n_layers)
        self.decoder = Decoder(n_latents, z_dim, h_dim, hid_dim, x_dim, n_layers)

    def forward(self, x):
        hs = self.encoder(x)
        x_mu, x_logvar, kls, x_recon = self.decoder(hs)

        x_in = x.reshape(-1, self.x_dim)
        nll = gaussian_nll(x_in, x_mu, x_logvar).sum() / x.shape[0]

        elbo = nll + 0.
        for kl in kls:
            kl /= x.shape[0]
            elbo += kl
        
        return elbo, nll, OrderedDict({f'KL_z{len(kls) - i}': kls[i] for i in range(len(kls))}), x_recon.reshape(x.shape)
    
    def sample(self, n_samples, n_points_per_cloud_gen, device='cuda'):
        samples = self.decoder.sample(n_samples, n_points_per_cloud_gen, device=device)
        return samples


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(10, 5, 3).to(device)
    model = SingleShapeDeepVAE(4, 3, 8, 8, 2, 3).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    elbo, nll, kls, x_recon = model(x)
    print(elbo, nll, kls, x_recon.shape)

    optimizer.zero_grad()
    elbo.backward()
    optimizer.step()

    with torch.no_grad():
        samples = model.sample(2, 100)
        print(samples.shape)