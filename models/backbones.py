from random import Random
from tkinter import X
import torch
import torch.nn as nn
from models.lipschitzLinear import LipschitzLinear
from siren_pytorch import SirenNet
from models.rff import RandomFF

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

net = SirenNet(
    dim_in = 2,                        # input dimension, ex. 2d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 2,                       # output dimension, ex. rgb value
    num_layers = 3,                    # number of layers
    final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

class SMLP(nn.Module):
    def __init__(self, in_dim, out_dim, c_dim, hid_dim=64, n_resnet_blocks=1, activation='relu', last_activation=None, 
                use_batchnorms=False, use_lipschitz_norm=False):
        super(SMLP, self).__init__()

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
        self.siren = SirenNet(dim_in=in_dim,dim_hidden= 256, dim_out= in_dim,num_layers=3, 
                                final_activation= nn.Sigmoid(), w0_initial=30.)
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
        #print(self.siren(x).size)
        if self.use_batchnorms:
            x= self.siren(x)
            x = self.activation(self.bn_in_x(self.fc_in_x(x)))
            c = self.activation(self.bn_in_c(self.fc_in_c(c)))
        else:
            x= self.siren(x)
            #print("after siren")
            #print(x.size())
            x = self.activation(self.fc_in_x(x))
            c = self.activation(self.fc_in_c(c))

        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, c=c)

        x = self.fc_out(torch.cat([x, c], dim=1))
        return self.last_activation(x)


class FMLP(nn.Module):
    def __init__(self, in_dim, out_dim, c_dim, hid_dim=64, n_resnet_blocks=1, activation='relu', last_activation=None, 
                use_batchnorms=False, use_lipschitz_norm=False):
        super(FMLP, self).__init__()

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
        self.rff = RandomFF(in_features= in_dim, fourier_features=hid_dim*4,
                 hidden_features=hid_dim*4, hidden_layers=2, out_features=out_dim, scale=10)
        
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
        #print(self.rff(x).size())
        if self.use_batchnorms:
            x= self.rff(x)
            x = self.activation(self.bn_in_x(self.fc_in_x(x)))
            c = self.activation(self.bn_in_c(self.fc_in_c(c)))
        else:
            x= self.rff(x)
            #print("after rff")
            #print(x.size())
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

        _x += x # TODO check if not torch.cat([_x, x], dim=1)
        return self.activation(_x)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(5, 2).to(device)
    c = torch.randn(5, 3).to(device)
    model = CMLP(2, 4, 3).to(device)

    print(model)

    x = model(x, c)

    print(x.shape)