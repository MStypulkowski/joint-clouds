import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class LipschitzLinear(nn.Module):
    # Based on https://arxiv.org/pdf/2202.08345.pdf
    def __init__(self, in_dim, out_dim):
        super(LipschitzLinear, self).__init__()

        self.W = Parameter(torch.empty((out_dim, in_dim)))
        self.b = Parameter(torch.empty(out_dim))
        self.c = Parameter(torch.empty(1))

        self.reset_parameters()

    def normalization(self, W, c):
        absrowsum = W.abs().sum(1)
        scale = torch.minimum(torch.tensor(1., device=W.device), c / absrowsum)
        return W * scale.unsqueeze(1)
        # return W
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

        with torch.no_grad():
            absrowsum = self.W.abs().sum(1)
            inf_norm = absrowsum.max()
            self.c.fill_(inf_norm)

    def forward(self, x):
        W_norm = self.normalization(self.W, self.c)
        return F.linear(x, W_norm, self.b)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    liplin = LipschitzLinear(3, 4).to(device)
    x = torch.randn(2, 4).to(device)

    y = liplin(x)

    print(x.shape, y.shape, liplin.c)