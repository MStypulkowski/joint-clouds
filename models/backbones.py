from tkinter import X
import torch
import torch.nn as nn
from math import pi as PI
from models.lipschitzLinear import LipschitzLinear


class PositionalEncoding:
    def __init__(self, L, x_dim):
        self.L = L
        self.x_dim = x_dim
    
    def encode(self, x):
        n = x.shape[0]
        x_expand = x.unsqueeze(-1).expand(n, self.x_dim, self.L).reshape(n, -1)
        power_mask = torch.tensor([2**i * PI for _ in range(self.x_dim) for i in range(-1, self.L - 1)]).reshape(1, -1).expand(n, -1).to(x.device)
        x_expand *= power_mask
        return torch.cat([torch.sin(x_expand), torch.cos(x_expand)], dim=1)

    def decode(self, y):
        y_sin = y[:, [i * self.L for i in range(self.x_dim)]]
        y_decoded = torch.asin(y_sin) / (PI / 2)
        return y_decoded


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, n_resnet_blocks=1, activation='relu', last_activation=None, 
                use_batchnorms=False, use_lipschitz_norm=False):
        super(MLP, self).__init__()

        if use_lipschitz_norm:
            base_layer = LipschitzLinear
        else:
            base_layer = nn.Linear

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        if last_activation is None:
            self.last_activation = lambda x: x
        elif last_activation == 'tanh':
            self.last_activation = nn.Tanh()

        self.fc_in = base_layer(in_dim, hid_dim)
        self.resnet_blocks = []
        for _ in range(n_resnet_blocks):
            self.resnet_blocks.append(ResNetBlock(hid_dim, base_layer=base_layer, activation=activation, use_batchnorms=use_batchnorms))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.fc_out = base_layer(hid_dim, out_dim)

        self.use_batchnorms = use_batchnorms
        if use_batchnorms:
            self.bn_in = nn.BatchNorm1d(hid_dim)

    def forward(self, x):
        assert x.dim() == 2

        if self.use_batchnorms:
            x = self.activation(self.bn_in(self.fc_in(x)))
        else:
            x = self.activation(self.fc_in(x))

        x = self.resnet_blocks(x)

        x = self.fc_out(x)
        return self.last_activation(x)


class CMLP(nn.Module):
    def __init__(self, in_dim, out_dim, c_dim, hid_dim=64, n_resnet_blocks=1, activation='relu', last_activation=None, 
                use_batchnorms=False, use_lipschitz_norm=False):
        super(CMLP, self).__init__()

        if use_lipschitz_norm:
            base_layer = LipschitzLinear
        else:
            base_layer = nn.Linear

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        if last_activation is None:
            self.last_activation = lambda x: x
        elif last_activation == 'tanh':
            self.last_activation = nn.Tanh()

        self.fc_in_x = base_layer(in_dim, hid_dim // 2)
        self.fc_in_c = base_layer(c_dim, hid_dim // 2)
        self.resnet_blocks = []
        for _ in range(n_resnet_blocks):
            self.resnet_blocks.append(ResNetBlock(hid_dim, conditional=True, base_layer=base_layer, activation=activation, use_batchnorms=use_batchnorms))
        self.resnet_blocks = nn.ModuleList(self.resnet_blocks)
        self.fc_out = base_layer(hid_dim, out_dim)

        self.use_batchnorms = use_batchnorms
        if use_batchnorms:
            self.bn_in_x = nn.BatchNorm1d(hid_dim // 2)
            self.bn_in_c = nn.BatchNorm1d(hid_dim // 2)

    def forward(self, x, c):
        assert x.dim() == 2

        if self.use_batchnorms:
            x = self.activation(self.bn_in_x(self.fc_in_x(x)))
            c = self.activation(self.bn_in_c(self.fc_in_c(c)))
        else:
            x = self.activation(self.fc_in_x(x))
            c = self.activation(self.fc_in_c(c))

        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, c=c)

        x = self.fc_out(torch.cat([x, c], dim=1))
        return self.last_activation(x)


class ResNetBlock(nn.Module):
    def __init__(self, hid_dim, conditional=False, base_layer=nn.Linear, activation='relu', use_batchnorms=False):
        super(ResNetBlock, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        if conditional:
            out_dim = hid_dim // 2
        else:
            out_dim = hid_dim

        self.fc1 = base_layer(hid_dim, out_dim)
        self.fc2 = base_layer(hid_dim, out_dim)

        self.use_batchnorms = use_batchnorms
        if use_batchnorms:
            self.bn1 = nn.BatchNorm1d(out_dim)
            self.bn2 = nn.BatchNorm1d(out_dim)

        # self.beta = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, x, c=None):
        if c is not None:
            _x = torch.cat([x, c], dim=1)
        else:
            _x = x

        if self.use_batchnorms:
            _x = self.activation(self.bn1(self.fc1(_x)))
            if c is not None:
                _x = torch.cat([_x, c], dim=1)
            _x = self.bn2(self.fc2(_x))
        else:
            _x = self.activation(self.fc1(_x))
            if c is not None:
                _x = torch.cat([_x, c], dim=1)
            _x = self.fc2(_x)


        # _x = self.beta * _x + x
        _x += x
        return self.activation(_x)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(5, 2).to(device)
    c = torch.randn(5, 3).to(device)
    model = CMLP(2, 4, 3).to(device)

    print(model)

    x = model(x, c)

    print(x.shape)