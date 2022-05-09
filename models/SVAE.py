import torch
import pdb
import torch.nn as nn
import torch.distributions
from models.utils import count_trainable_parameters, reparametrization, analytical_kl, gaussian_nll
from utils.von_mises_fisher import HypersphericalUniform 
from utils.von_mises_fisher import VonMisesFisher
from torch.distributions.kl import register_kl

class CLinear(nn.Module):
    def __init__(self, in_dim, c_dim, out_dim, activation=nn.SiLU(), last_activation=None):
        super(CLinear, self).__init__()
        self.fc_x = nn.Linear(in_dim, out_dim // 2)
        self.fc_c = nn.Linear(c_dim, out_dim // 2)
        self.fc = nn.Linear(out_dim, out_dim)
        self.activation = activation
        self.last_activation = last_activation

    def forward(self, x, c):
        x = self.fc_x(x)
        c = self.fc_c(c)
        xc = torch.cat([x, c], dim=1)
        xc = self.activation(xc)
        xc = self.fc(xc)
        return self.last_activation(xc) if self.last_activation else xc


class Block(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, activation=nn.SiLU(), last_activation=None):
        super(Block, self).__init__()
        assert n_layers > 1

        self.fcs = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 2):
            self.fcs.append(nn.Linear(hid_dim, hid_dim))
        self.fcs.append(nn.Linear(hid_dim, out_dim))
        self.fcs = nn.ModuleList(self.fcs)

        self.activation = activation
        self.last_activation = last_activation

    def forward(self, x):
        for fc in self.fcs:
            x = self.activation(fc(x))
        return self.last_activation(x) if self.last_activation else x


class CBlock(nn.Module):
    def __init__(self, in_dim, c_dim, hid_dim, out_dim, n_layers, activation=nn.SiLU(), last_activation=None):
        super(CBlock, self).__init__()
        assert n_layers > 1

        self.fcs = [CLinear(in_dim, c_dim, hid_dim, activation=activation, last_activation=activation)]
        for _ in range(n_layers - 2):
            self.fcs.append(CLinear(hid_dim, c_dim, hid_dim, activation=activation, last_activation=activation))
        self.fcs.append(CLinear(hid_dim, c_dim, out_dim, activation=activation, last_activation=last_activation))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x, c):
        for fc in self.fcs:
            x = fc(x, c)
        return x
        

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, emb_dim, hid_dim, n_layers, activation=nn.SiLU(), last_activation=None):
        super(Encoder, self).__init__()

        self.nn_x_h = Block(x_dim, hid_dim, h_dim, n_layers, activation=activation, last_activation=last_activation)
        self.nn_h_e = Block(h_dim, hid_dim, emb_dim, n_layers, activation=activation, last_activation=last_activation)
        
        self.nn_h_z = Block(h_dim, hid_dim, z_dim, n_layers, activation=activation, last_activation=last_activation)
        self.nn_h_z_k = Block(h_dim, hid_dim, 1, n_layers, activation=activation, last_activation=last_activation)
        self.nn_e_ze = Block(emb_dim, hid_dim, 2 * emb_dim, n_layers, activation=activation, last_activation=last_activation)

    def forward(self, x):
        n_clouds, n_points, x_dim = x.shape[0], x.shape[1], x.shape[2]

        x = x.reshape(-1, x_dim) # N*M x x_dim
        h = self.nn_x_h(x) # N*M x h_dim

        e = h.reshape(n_clouds, n_points, -1) # N x M x h_dim
        e = torch.max(e, 1, keepdim=True)[0] # N x 1 x h_dim
        e = e.squeeze(1) # N x h_dim
        e = self.nn_h_e(e) # N x emb_dim

        z_mu= self.nn_h_z(h) + .00001
        #print(z_mu.norm(dim=-1, keepdim=True))
        z_mu= z_mu / (z_mu.norm(dim=-1, keepdim=True) + .0001)

        z_logvar= self.nn_h_z_k(h) + 1
        ze_mu, ze_logvar = self.nn_e_ze(e).chunk(2, 1) # N x emb_dim

        return z_mu, z_logvar, ze_mu, ze_logvar


class Decoder(nn.Module):
    def __init__(self, z_dim, emb_dim, x_dim, hid_dim, n_layers, activation=nn.SiLU(), last_activation=None):
        super(Decoder, self).__init__()
        self.nn_z_x = CBlock(z_dim, emb_dim, hid_dim, 2 * x_dim, n_layers, activation=activation, last_activation=last_activation)

    def forward(self, z, ze):
        ze = ze.unsqueeze(1).expand(-1, z.shape[0] // ze.shape[0], ze.shape[-1]).reshape(-1, ze.shape[-1]) # z.shape[0] // ze.shape[0] gives number of points
        x_mu, x_logvar = self.nn_z_x(z, ze).chunk(2, 1)

        return x_mu, x_logvar


class S_VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, emb_dim, encoder_hid_dim, encoder_n_layers, decoder_hid_dim, decoder_n_layers, activation=nn.SiLU(), last_activation=None):
        super(S_VAE, self).__init__()

        self.z_dim = z_dim
        self.emb_dim = emb_dim

        self.encoder = Encoder(x_dim, h_dim, z_dim, emb_dim, encoder_hid_dim, encoder_n_layers, activation=activation, last_activation=last_activation)
        self.decoder = Decoder(z_dim, emb_dim, x_dim, decoder_hid_dim, decoder_n_layers, activation=activation, last_activation=last_activation)

    def forward(self, x):
        z_mu, z_logvar, ze_mu, ze_logvar = self.encoder(x)
        if z_mu.isnan().any():
            pdb.set_trace()
        if z_logvar.isnan().any():
            pdb.set_trace()
        if ze_mu.isnan().any():
            pdb.set_trace()
        if ze_logvar.isnan().any():
            pdb.set_trace()
        
        #z = reparametrization(z_mu, z_logvar)
        q_z = VonMisesFisher(z_mu, z_logvar)
        
        
        p_z = HypersphericalUniform(self.z_dim - 1)
        z = q_z.rsample()
    
        ze = reparametrization(ze_mu, ze_logvar)
        if ze.isnan().any():
            pdb.set_trace()
        if z.isnan().any():
            pdb.set_trace()
        x_mu, x_logvar = self.decoder(z, ze)

        
        kl_z= torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        
        #kl_z = analytical_kl(z_mu, torch.zeros_like(z_mu), z_logvar, torch.zeros_like(z_logvar)).sum() / x.shape[0]
        kl_ze = analytical_kl(ze_mu, torch.zeros_like(ze_mu), ze_logvar, torch.zeros_like(ze_logvar)).sum() / x.shape[0]
        print('x_logvar is', x_logvar)
        if x_logvar.isnan().any():
            pdb.set_trace()
        nll = gaussian_nll(x.reshape(-1, x.shape[-1]), x_mu, x_logvar).sum() / x.shape[0]

        x_recon = reparametrization(x_mu, x_logvar).reshape(x.shape)

        return nll, kl_ze, kl_z, x_recon

    
    def sample(self, n_clouds, n_points, x_dim, device='cuda'):
        z = reparametrization(torch.zeros(n_clouds * n_points, self.z_dim).to(device), torch.zeros(n_clouds * n_points, self.z_dim).to(device))
        ze = reparametrization(torch.zeros(n_clouds, self.emb_dim).to(device), torch.zeros(n_clouds, self.emb_dim).to(device))
        x_mu, x_logvar = self.decoder(z, ze)
        x_sample = reparametrization(x_mu, x_logvar).reshape(n_clouds, n_points, x_dim)

        return x_sample


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(3, 5, 2).to(device)
    model = S_VAE(2, 32, 2, 16, 64, 2, 128, 4).to(device)
    print(model)
    print(count_trainable_parameters(model))

    nll, kl_ze, kl_z, x_recon = model(x)
    print(nll, kl_ze, kl_z, x_recon.shape)

    samples = model.sample(3, 100, 2, device=device)
    print(samples.shape)
