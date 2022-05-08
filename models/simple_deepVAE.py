import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import count_trainable_parameters, reparametrization, analytical_kl, gaussian_nll


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
        

class HyperNet(nn.Module):
    def __init__(self, target_in_dim, target_hid_dim, target_out_dim, target_n_layers, 
                hyper_in_dim, hyper_hid_dim, hyper_n_layers,
                activation=nn.SiLU()):
        super(HyperNet, self).__init__()
        self.target_in_dim = target_in_dim
        self.target_hid_dim = target_hid_dim
        self.target_out_dim = target_out_dim
        self.target_n_layers = target_n_layers
        self.activation = activation

        self.hypernet_in = Block(hyper_in_dim, hyper_hid_dim, target_in_dim * target_hid_dim + target_hid_dim, hyper_n_layers, activation=activation, last_activation=None)
        self.hypernet_hid = CBlock(hyper_in_dim, target_n_layers - 2, hyper_hid_dim, target_hid_dim * target_hid_dim + target_hid_dim, hyper_n_layers, activation=activation, last_activation=None)
        self.hypernet_out = Block(hyper_in_dim, hyper_hid_dim, target_hid_dim * target_out_dim + target_out_dim, hyper_n_layers, activation=activation, last_activation=None)

    def forward(self, x, c):
        weights = self.hypernet_in(c)
        x = self.forward_layer(x, weights, self.target_in_dim, self.target_hid_dim)

        for i in range(self.target_n_layers - 2):
            # layer_oh = F.one_hot(torch.tensor(i), self.target_n_layers - 2).float().to(c.device)
            # layer_oh.unsquueze(-1).expand(c.shape[0])
            layer_oh = F.one_hot(torch.ones(c.shape[0], dtype=int) * i, self.target_n_layers - 2).float().to(c.device)
            weights = self.hypernet_hid(c, layer_oh)
            x = self.forward_layer(x, weights, self.target_hid_dim, self.target_hid_dim)

        weights = self.hypernet_out(c)
        x = self.forward_layer(x, weights, self.target_hid_dim, self.target_out_dim, activation=False)

        return x

    def forward_layer(self, x, weights, in_dim, out_dim, activation=True):
        w, b = weights[:, :in_dim * out_dim].reshape(-1, out_dim, in_dim), weights[:, in_dim * out_dim:]
        x = torch.matmul(w, x.unsqueeze(-1)).squeeze(-1) + b
        return self.activation(x) if activation else x


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, emb_dim, hid_dim, n_layers, activation=nn.SiLU(), last_activation=None):
        super(Encoder, self).__init__()

        self.nn_x_h = Block(x_dim, hid_dim, h_dim, n_layers, activation=activation, last_activation=last_activation)
        self.nn_h_e = Block(h_dim, hid_dim, emb_dim, n_layers, activation=activation, last_activation=last_activation)
        
        self.nn_h_z = Block(h_dim, hid_dim, 2 * z_dim, n_layers, activation=activation, last_activation=last_activation)
        self.nn_e_ze = Block(emb_dim, hid_dim, 2 * emb_dim, n_layers, activation=activation, last_activation=last_activation)

    def forward(self, x):
        n_clouds, n_points, x_dim = x.shape[0], x.shape[1], x.shape[2]

        x = x.reshape(-1, x_dim) # N*M x x_dim
        h = self.nn_x_h(x) # N*M x h_dim

        e = h.reshape(n_clouds, n_points, -1) # N x M x h_dim
        e = torch.max(e, 1, keepdim=True)[0] # N x 1 x h_dim
        e = e.squeeze(1) # N x h_dim
        e = self.nn_h_e(e) # N x emb_dim

        z_mu, z_logvar = self.nn_h_z(h).chunk(2, 1) # N*M x z_dim
        ze_mu, ze_logvar = self.nn_e_ze(e).chunk(2, 1) # N x emb_dim

        return z_mu, z_logvar, ze_mu, ze_logvar


class Decoder(nn.Module):
    def __init__(self, z_dim, emb_dim, x_dim, hid_dim, n_layers, activation=nn.SiLU(), last_activation=None, use_hypernet=False, hyper_hid_dim=None, hyper_n_layers=None):
        super(Decoder, self).__init__()
        self.use_hypernet = use_hypernet

        if use_hypernet:
            assert hyper_hid_dim is not None and hyper_n_layers is not None
            self.nn_z_x = HyperNet(z_dim, hid_dim, 2 * x_dim, n_layers, emb_dim, hyper_hid_dim, hyper_n_layers, activation=activation)
        else:
            self.nn_z_x = CBlock(z_dim, emb_dim, hid_dim, 2 * x_dim, n_layers, activation=activation, last_activation=last_activation)

    def forward(self, z, ze):
        ze = ze.unsqueeze(1).expand(-1, z.shape[0] // ze.shape[0], ze.shape[-1]).reshape(-1, ze.shape[-1]) # z.shape[0] // ze.shape[0] gives number of points
        x_mu, x_logvar = self.nn_z_x(z, ze).chunk(2, 1)

        return x_mu, x_logvar


class SimpleVAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, emb_dim, encoder_hid_dim, encoder_n_layers, decoder_hid_dim, decoder_n_layers, 
                activation=nn.SiLU(), last_activation=None, use_hypernet=False, hyper_hid_dim=None, hyper_n_layers=None):
        super(SimpleVAE, self).__init__()

        self.z_dim = z_dim
        self.emb_dim = emb_dim

        self.encoder = Encoder(x_dim, h_dim, z_dim, emb_dim, encoder_hid_dim, encoder_n_layers, activation=activation, last_activation=last_activation)
        self.decoder = Decoder(z_dim, emb_dim, x_dim, decoder_hid_dim, decoder_n_layers, 
                            activation=activation, last_activation=last_activation, use_hypernet=use_hypernet, hyper_hid_dim=hyper_hid_dim, hyper_n_layers=hyper_n_layers)

    def forward(self, x):
        z_mu, z_logvar, ze_mu, ze_logvar = self.encoder(x)
        z = reparametrization(z_mu, z_logvar)
        ze = reparametrization(ze_mu, ze_logvar)
        x_mu, x_logvar = self.decoder(z, ze)

        kl_z = analytical_kl(z_mu, torch.zeros_like(z_mu), z_logvar, torch.zeros_like(z_logvar)).sum() / x.shape[0]
        kl_ze = analytical_kl(ze_mu, torch.zeros_like(ze_mu), ze_logvar, torch.zeros_like(ze_logvar)).sum() / x.shape[0]

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
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 3721))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(3, 5, 2).to(device)
    model = SimpleVAE(2, 32, 2, 16, 64, 2, 128, 4, use_hypernet=True, hyper_hid_dim=256, hyper_n_layers=4).to(device)
    print(model)
    print(count_trainable_parameters(model))

    nll, kl_ze, kl_z, x_recon = model(x)
    print(nll, kl_ze, kl_z, x_recon.shape)

    samples = model.sample(3, 100, 2, device=device)
    print(samples.shape)
