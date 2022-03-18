import torch
import torch.nn as nn
from models.lipschitzLinear import LipschitzLinear


class PointNet(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, n_layers=3, activation='relu', last_activation=None,
                 kernel_size=1, pooling=False, use_batchnorms=True, use_lipschitz_norm=True):
        super(PointNet, self).__init__()
        assert n_layers >= 2
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        
        self.last_activation = None
        if last_activation == 'tanh':
            self.last_activation = nn.Tanh()

        self.pooling = pooling

        self.use_batchnorms = use_batchnorms

        self.convs = [nn.Conv1d(in_dim, hid_dim, kernel_size)]
        if use_batchnorms:
            self.bns = [nn.BatchNorm1d(hid_dim)]
        for _ in range(n_layers - 2):
            self.convs.append(nn.Conv1d(hid_dim, hid_dim, kernel_size))
            if use_batchnorms:
                self.bns.append(nn.BatchNorm1d(hid_dim))
        self.convs.append(nn.Conv1d(hid_dim, out_dim, kernel_size))
        self.convs = nn.ModuleList(self.convs)
        if use_batchnorms:
            self.bns = nn.ModuleList(self.bns)

        if pooling:
            # TODO
            # Add pooling and fcs
            raise NotImplementedError

    def forward(self, x):
        assert x.dim() == 3

        if self.use_batchnorms:
            for fc, bn in zip(self.convs[:-1], self.bns):
                x = self.activation(bn(fc(x)))
        else:
            for fc in self.convs[:-1]:
                x = self.activation(fc(x))
        x = self.convs[-1](x)

        if self.pooling:
            pass

        if self.last_activation is not None:
            x = self.last_activation(x)

        return x


class CPointNet(nn.Module):
    def __init__(self, in_dim, out_dim, c_dim, hid_dim=64, n_layers=3, activation='relu', last_activation=None, 
                kernel_size=1, pooling=False, use_batchnorms=True, use_lipschitz_norm=True):
        super(CPointNet, self).__init__()
        assert n_layers >= 2

        self.c_dim = c_dim

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        self.last_activation = None
        if last_activation == 'tanh':
            self.last_activation = nn.Tanh()

        self.pooling = pooling

        self.use_batchnorms = use_batchnorms

        self.conv_x = nn.Conv1d(in_dim, hid_dim // 2, kernel_size)
        self.conv_c = nn.Conv1d(c_dim, hid_dim // 2, kernel_size)

        self.convs = []
        if use_batchnorms:
            self.bns = [nn.BatchNorm1d(hid_dim)]
        for _ in range(n_layers - 2):
            self.convs.append(nn.Conv1d(hid_dim, hid_dim, kernel_size))
            if use_batchnorms:
                self.bns.append(nn.BatchNorm1d(hid_dim))
        self.convs.append(nn.Conv1d(hid_dim, out_dim, kernel_size))
        self.convs = nn.ModuleList(self.convs)
        if use_batchnorms:
            self.bns = nn.ModuleList(self.bns)

        if pooling:
            # TODO
            # Add pooling and fcs
            raise NotImplementedError

    def forward(self, x, c):
        assert x.dim() == 3
        
        x = self.conv_x(x)
        c = self.conv_c(c)

        xc = torch.cat([x, c], dim=1)
        if self.use_batchnorms:
            for fc, bn in zip(self.convs[:-1], self.bns):
                xc = self.activation(bn(fc(xc)))
        else:
            for fc in self.convs[:-1]:
                xc = self.activation(fc(xc))
        xc = self.convs[-1](xc)

        if self.pooling:
            pass
        
        if self.last_activation is not None:
            xc = self.last_activation(xc)

        return xc


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, n_layers=3, activation='relu', last_activation=None, 
                use_batchnorms=False, use_lipschitz_norm=True):
        super(MLP, self).__init__()
        assert n_layers >= 2

        if use_lipschitz_norm:
            base_layer = LipschitzLinear
        else:
            base_layer = nn.Linear

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        self.last_activation = None
        if last_activation == 'tanh':
            self.last_activation = nn.Tanh()

        self.use_batchnorms = use_batchnorms

        self.fcs = [base_layer(in_dim, hid_dim)]
        if use_batchnorms:
            self.bns = [nn.BatchNorm1d(hid_dim)]
        for _ in range(n_layers - 2):
            self.fcs.append(base_layer(hid_dim, hid_dim))
            if use_batchnorms:
                self.bns.append(nn.BatchNorm1d(hid_dim))
        self.fcs.append(base_layer(hid_dim, out_dim))
        self.fcs = nn.ModuleList(self.fcs)
        if use_batchnorms:
            self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        assert x.dim() == 2

        if self.use_batchnorms:
            for fc, bn in zip(self.fcs[:-1], self.bns):
                x = self.activation(bn(fc(x)))
        else:
            for fc in self.fcs[:-1]:
                x = self.activation(fc(x))
        x = self.fcs[-1](x)

        if self.last_activation is not None:
            x = self.last_activation(x)

        return x


class CMLP(nn.Module):
    def __init__(self, in_dim, out_dim, c_dim, hid_dim=64, n_layers=3, activation='relu', last_activation=None, 
                use_batchnorms=False, use_lipschitz_norm=True):
        super(CMLP, self).__init__()
        assert n_layers >= 2

        if use_lipschitz_norm:
            base_layer = LipschitzLinear
        else:
            base_layer = nn.Linear

        self.c_dim = c_dim

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        self.last_activation = None
        if last_activation == 'tanh':
            self.last_activation = nn.Tanh()

        self.use_batchnorms = use_batchnorms

        self.fc_x = base_layer(in_dim, hid_dim // 2)
        self.fc_c = base_layer(c_dim, hid_dim // 2)

        self.fcs = []
        if use_batchnorms:
            self.bns = [nn.BatchNorm1d(hid_dim)]
        for _ in range(n_layers - 2):
            self.fcs.append(base_layer(hid_dim, hid_dim))
            if use_batchnorms:
                self.bns.append(nn.BatchNorm1d(hid_dim))
        self.fcs.append(base_layer(hid_dim, out_dim))
        self.fcs = nn.ModuleList(self.fcs)
        if use_batchnorms:
            self.bns = nn.ModuleList(self.bns)

    def forward(self, x, c):
        assert x.dim() == 2

        x = self.fc_x(x)
        c = self.fc_c(c)
        
        # xc = torch.cat([x, c.unsqueeze(-1).expand(-1, self.c_dim, x.shape[-1])], dim=1)
        xc = torch.cat([x, c], dim=1)

        if self.use_batchnorms:
            for fc, bn in zip(self.fcs[:-1], self.bns):
                xc = self.activation(bn(fc(xc)))
        else:
            for fc in self.fcs[:-1]:
                xc = self.activation(fc(xc))
        xc = self.fcs[-1](xc)

        if self.last_activation is not None:
            xc = self.last_activation(xc)

        return xc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # x = torch.randn(10, 2, 100).to(device)
    # c = torch.randn(10, 5).to(device)
    # model = CPointNet(2, 16, 5, pooling=False).to(device)

    x = torch.randn(5, 2).to(device)
    c = torch.randn(5, 3).to(device)
    model = CMLP(2, 4, 3).to(device)

    print(model)

    x = model(x, c)

    print(x.shape)