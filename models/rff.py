import torch
import torch.nn as nn
import math


class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        B = torch.randn(in_features, out_features)*scale
        self.register_buffer("B", B)
    
    def forward(self, x):
        #print("x is",x)
        x_proj = torch.matmul(2*math.pi*x, self.B)
        #print(x_proj)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        #print('out is',out)
        return out

class RandomFF(nn.Module):
    def __init__(self, in_features, fourier_features,
                 hidden_features, hidden_layers, out_features, scale):
        super().__init__()

        self.net = []
        if fourier_features is not None:
            #print("in features", in_features)
            self.net.append(FourierLayer(in_features, fourier_features, scale))
            self.net.append(nn.Linear(2*fourier_features, hidden_features))
            self.net.append(nn.ReLU())
        else:
            self.net.append(nn.Linear(in_features, hidden_features))
            self.net.append(nn.ReLU())
        
        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        out = self.net(x)
        return out