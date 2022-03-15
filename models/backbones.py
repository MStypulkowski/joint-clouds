import torch
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, n_layers=3, activation='relu', last_activation=None, kernel_size=1, pooling=False):
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

        self.convs = [nn.Conv1d(in_dim, hid_dim, kernel_size)]
        for _ in range(n_layers - 2):
            self.convs.append(nn.Conv1d(hid_dim, hid_dim, kernel_size))
        self.convs.append(nn.Conv1d(hid_dim, out_dim, kernel_size))
        self.convs = nn.ModuleList(self.convs)

        if pooling:
            # TODO
            # Add pooling and fcs
            raise NotImplementedError

    def forward(self, x):
        assert x.dim() == 3

        for fc in self.convs[:-1]:
            x = self.activation(fc(x))
        x = self.convs[-1](x)

        if self.pooling:
            pass

        if self.last_activation is not None:
            x = self.last_activation(x)

        return x


class CPointNet(nn.Module):
    def __init__(self, in_dim, out_dim, c_dim, hid_dim=64, n_layers=3, activation='relu', last_activation=None, kernel_size=1, pooling=False):
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

        self.convs = [nn.Conv1d(in_dim + c_dim, hid_dim, kernel_size)]
        for _ in range(n_layers - 2):
            self.convs.append(nn.Conv1d(hid_dim, hid_dim, kernel_size))
        self.convs.append(nn.Conv1d(hid_dim, out_dim, kernel_size))
        self.convs = nn.ModuleList(self.convs)

        if pooling:
            # TODO
            # Add pooling and fcs
            raise NotImplementedError

    def forward(self, x, c):
        assert x.dim() == 3

        # xc = torch.cat([x, c.unsqueeze(-1).expand(-1, self.c_dim, x.shape[-1])], dim=1)
        xc = torch.cat([x, c], dim=1)
        for fc in self.convs[:-1]:
            xc = self.activation(fc(xc))
        xc = self.convs[-1](xc)

        if self.pooling:
            pass
        
        if self.last_activation is not None:
            xc = self.last_activation(xc)

        return xc


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, n_layers=3, activation='relu', last_activation=None):
        super(MLP, self).__init__()
        assert n_layers >= 2
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        self.last_activation = None
        if last_activation == 'tanh':
            self.last_activation = nn.Tanh()

        self.fcs = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 2):
            self.fcs.append(nn.Linear(hid_dim, hid_dim))
        self.fcs.append(nn.Linear(hid_dim, out_dim))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        assert x.dim() == 2

        for fc in self.fcs[:-1]:
            x = self.activation(fc(x))

        x = self.fcs[-1](x)

        if self.last_activation is not None:
            x = self.last_activation(x)

        return x


class CMLP(nn.Module):
    def __init__(self, in_dim, out_dim, c_dim, hid_dim=64, n_layers=3, activation='relu', last_activation=None):
        super(CMLP, self).__init__()
        assert n_layers >= 2

        self.c_dim = c_dim

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        self.last_activation = None
        if last_activation == 'tanh':
            self.last_activation = nn.Tanh()

        self.fcs = [nn.Linear(in_dim + c_dim, hid_dim)]
        for _ in range(n_layers - 2):
            self.fcs.append(nn.Linear(hid_dim, hid_dim))
        self.fcs.append(nn.Linear(hid_dim, out_dim))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x, c):
        assert x.dim() == 2

        # xc = torch.cat([x, c.unsqueeze(-1).expand(-1, self.c_dim, x.shape[-1])], dim=1)
        xc = torch.cat([x, c], dim=1)

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

    x = torch.randn(10, 16).to(device)
    c = torch.randn(10, 5).to(device)
    model = CMLP(16, 32, 5).to(device)

    print(model)

    x = model(x, c)

    print(x.shape)