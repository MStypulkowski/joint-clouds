from collections import OrderedDict
import torch
import torch.nn as nn

from models.utils import reparametrization, analytical_kl, gaussian_nll


class CLinear(nn.Module):
    def __init__(self, in_dim, c_dim, hid_dim):
        super(CLinear, self).__init__()
        self.fc = nn.Linear(in_dim + c_dim, hid_dim)
    def forward(self, x, c):
        xc = torch.cat([x, c], dim=1) # TODO preprocessing
        return self.fc(xc)


class Block(nn.Module):
    def __init__(self, in_dim, c_dim, hid_dim, out_dim, residual=True):
        super(Block, self).__init__()

        if residual:
            assert in_dim == out_dim

        self.residual = residual

        if c_dim is None:
            self.fc1 = nn.Linear(in_dim, hid_dim)
            self.fc2 = nn.Linear(hid_dim, hid_dim)
            self.fc3 = nn.Linear(hid_dim, out_dim)    
        else:
            self.fc1 = CLinear(in_dim, c_dim, hid_dim)
            self.fc2 = CLinear(hid_dim, c_dim, hid_dim)
            self.fc3 = CLinear(hid_dim, c_dim, out_dim)    

        self.activation = nn.SiLU()

    def forward(self, x, c=None):
        if c is None:
            _x = self.activation(self.fc1(x))
            _x = self.activation(self.fc2(_x))
            _x = self.activation(self.fc3(_x))
        else:
            _x = self.activation(self.fc1(x, c))
            _x = self.activation(self.fc2(_x, c))
            _x = self.activation(self.fc3(_x, c))
        # TODO out activation
        return _x + x if self.residual else _x


class TopDownBlock(nn.Module):
    def __init__(self, z_dim, h_dim, ze_dim, hid_dim):
        super(TopDownBlock, self).__init__()

        self.q_block = Block(z_dim + h_dim, ze_dim, hid_dim, 2 * z_dim, residual=False)
        self.p_block = Block(z_dim, ze_dim, hid_dim, 3 * z_dim, residual=False)

        self.out_block = Block(z_dim, ze_dim, hid_dim, z_dim, residual=True)

        self.fc_z = CLinear(z_dim, ze_dim, z_dim)
        # TODO activation

    def forward(self, z, h, ze):
        zh = torch.cat([z, h], dim=1)
        q_mu, q_logvar = self.q_block(zh, ze).chunk(2, 1)

        p_out = self.p_block(z, ze)
        z_residual, (p_mu, p_logvar) = p_out[:, :z.shape[-1]], p_out[:, z.shape[-1]:].chunk(2, 1)

        z_sample = reparametrization(q_mu, q_logvar)
        z_sample = self.fc_z(z_sample, ze)

        z = z + z_residual + z_sample
        z = self.out_block(z, ze)

        kl = analytical_kl(q_mu, p_mu, q_logvar, p_logvar)

        assert z.shape == kl.shape

        return z, kl.sum()
    
    def sample(self, z, ze):
        p_out = self.p_block(z, ze)
        z_residual, (p_mu, p_logvar) = p_out[:, :z.shape[-1]], p_out[:, z.shape[-1]:].chunk(2, 1)

        z_sample = reparametrization(p_mu, p_logvar)
        z_sample = self.fc_z(z_sample, ze)

        z = z + z_residual + z_sample
        z = self.out_block(z, ze)

        return z


class PriorBlock(nn.Module):
    # Block for z_e and z_n
    def __init__(self, z_dim, h_dim, ze_dim, hid_dim):
        super(PriorBlock, self).__init__()

        self.z_dim = z_dim

        self.q_block = Block(h_dim, ze_dim, hid_dim, 2 * z_dim, residual=False)

        self.out_block = Block(z_dim, ze_dim, hid_dim, z_dim, residual=True)

        if ze_dim is None:
            self.fc_z = nn.Linear(z_dim, z_dim)
        else:
            self.fc_z = CLinear(z_dim, ze_dim, z_dim)
        # TODO check if activation

    def forward(self, h, ze=None):
        q_mu, q_logvar = self.q_block(h, ze).chunk(2, 1)
        z_sample = reparametrization(q_mu, q_logvar)

        if ze is None:
            z_sample = self.fc_z(z_sample)
        else:
            z_sample = self.fc_z(z_sample, ze)

        z = self.out_block(z_sample, ze)

        kl = analytical_kl(q_mu, torch.zeros_like(q_mu), q_logvar, torch.zeros_like(q_logvar))

        return z, kl.sum()

    def sample(self, n_samples, ze=None, device='cuda'):
        z_sample = torch.randn(n_samples, self.z_dim).to(device)

        if ze is None:
            z_sample = self.fc_z(z_sample)
        else:
            z_sample = self.fc_z(z_sample, ze)

        z = self.out_block(z_sample, ze)

        return z


class Encoder(nn.Module):
    def __init__(self, n_latents, x_dim, h_dim, hid_dim, e_dim, ze_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        self.in_block = Block(x_dim, None, hid_dim, h_dim, residual=False)
        self.h_blocks = nn.ModuleList([Block(h_dim, None, hid_dim, h_dim, residual=True) for _ in range(n_latents)])
        self.h_cond_blocks = nn.ModuleList([Block(h_dim, ze_dim, hid_dim, h_dim, residual=True) for _ in range(n_latents)])
        self.e_block = Block(h_dim, None, hid_dim, e_dim, residual=False)

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
        e = self.e_block(e)

        return hs, e

    def condition(self, hs, ze):
        hs_cond = []
        for h_cond_block, h in zip(self.h_cond_blocks, hs):
            h_cond = h_cond_block(h, ze)
            hs_cond.append(h_cond)
        
        return hs_cond


class Decoder(nn.Module):
    def __init__(self, n_latents, ze_dim, e_dim, z_dim, h_dim, hid_dim, x_dim, n_points_per_cloud):
        super(Decoder, self).__init__()

        self.ze_dim = ze_dim
        self.n_points_per_cloud = n_points_per_cloud

        self.ze_block = PriorBlock(ze_dim, e_dim, None, hid_dim)
        self.zn_block = PriorBlock(z_dim, h_dim, ze_dim, hid_dim)
        self.z_blocks = nn.ModuleList([TopDownBlock(z_dim, h_dim, ze_dim, hid_dim) for _ in range(n_latents - 1)])
        self.x_block = Block(z_dim, ze_dim, hid_dim, 2 * x_dim, residual=False)

    def forward(self, hs, ze):
        z, kl_zn = self.zn_block(hs[-1], ze)

        kls = [kl_zn]
        for i, z_block in enumerate(self.z_blocks):
            z, kl = z_block(z, hs[-i - 1], ze)
            kls.append(kl)
        
        x_mu, x_logvar = self.x_block(z, ze).chunk(2, 1)
        x_recon = reparametrization(x_mu, x_logvar)

        return x_mu, x_logvar, kls, x_recon

    def sample(self, n_samples, n_points_per_cloud_gen, device='cuda'):
        ze = self.ze_block.sample(n_samples, device=device)
        ze = ze.unsqueeze(1).expand(-1, n_points_per_cloud_gen, self.ze_dim).reshape(-1, self.ze_dim)

        z = self.zn_block.sample(n_samples * n_points_per_cloud_gen, ze=ze, device=device)
        for z_block in self.z_blocks:
            z = z_block.sample(z, ze)

        x_mu, x_logvar = self.x_block(z, ze).chunk(2, 1)

        return reparametrization(x_mu, x_logvar).reshape(n_samples, n_points_per_cloud_gen, -1)

    def get_ze(self, e):
        ze, kl_ze = self.ze_block(e)
        ze = ze.unsqueeze(1).expand(-1, self.n_points_per_cloud, self.ze_dim).reshape(-1, self.ze_dim)
        return ze, kl_ze


class DeepVAE(nn.Module):
    def __init__(self, n_latents, x_dim, h_dim, hid_dim, e_dim, ze_dim, z_dim, n_points_per_cloud):
        super(DeepVAE, self).__init__()
        self.x_dim = x_dim
        self.encoder = Encoder(n_latents, x_dim, h_dim, hid_dim, e_dim, ze_dim)
        self.decoder = Decoder(n_latents, ze_dim, e_dim, z_dim, h_dim, hid_dim, x_dim, n_points_per_cloud)

    def forward(self, x):
        hs, e = self.encoder(x)
        ze, kl_ze = self.decoder.get_ze(e)
        hs_cond = self.encoder.condition(hs, ze)
        x_mu, x_logvar, kls, x_recon = self.decoder(hs_cond, ze)

        nll = gaussian_nll(x.reshape(-1, self.x_dim), x_mu, x_logvar).sum() / x.shape[0]

        elbo = nll + kl_ze
        for kl in kls:
            kl /= x.shape[0]
            elbo += kl
        
        return elbo, nll, kl_ze, OrderedDict({f'KL_z{len(kls) - i}': kls[i] for i in range(len(kls))}), x_recon.reshape(x.shape)
    
    def sample(self, n_samples, n_points_per_cloud_gen, device='cuda'):
        samples = self.decoder.sample(n_samples, n_points_per_cloud_gen, device=device)
        return samples


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(3, 5, 2).to(device)
    model = DeepVAE(3, 2, 8, 16, 8, 8, 2, 5).to(device)
    elbo, nll, kl_ze, kls, x_recon = model(x)
    print(elbo, nll, kl_ze, kls, x_recon.shape)

    samples = model.sample(2, 100)
    print(samples.shape)
