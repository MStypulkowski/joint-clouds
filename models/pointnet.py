import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, n_layers=3, activation='relu', kernel_size=1, pooling=False):
        super(PointNet, self).__init__()
        assert n_layers >= 2

        if activation == 'relu':
            self.activation = nn.ReLU()

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
        print('x', x.shape)


        # for fc in self.convs[:-1]:
        #     x = self.activation(fc(x))
        # x = self.convs[-1](x)

        # if self.pooling:
        #     pass

        return x


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hid_dim=64, n_layers=3, activation='relu'):
        super(MLP, self).__init__()
        assert n_layers >= 2
        if activation == 'relu':
            self.activation = nn.ReLU()

        self.fcs = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 2):
            self.fcs.append(nn.Linear(hid_dim, hid_dim))
        self.fcs.append(nn.Linear(hid_dim, n_classes))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        assert x.dim() == 2
        print('x-MLP', x.shape)
        # for fc in self.fcs[:-1]:
        #     x = self.activation(fc(x))

        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(10, 2, 100).to(device)
    pointnet = PointNet(2, 16, pooling=False).to(device)

    print(pointnet)

    x = pointnet(x)

    print(x.shape)