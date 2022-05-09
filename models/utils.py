import numpy as np
import torch


def count_trainable_parameters(model):
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.dim() == 1:
                count += len(p)
            elif p.dim() == 2:
                count += p.shape[0] * p.shape[1]
            else:
                count += p.shape[0] * p.shape[1] * p.shape[2]
    
    return count

    
def reparametrization(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std




def analytical_kl(mu1, mu2, logvar1, logvar2):
    return -0.5 + logvar2 - logvar1 + 0.5 * (logvar1.exp() ** 2 + (mu1 - mu2) ** 2) / (logvar2.exp() ** 2)


def gaussian_nll(x, x_mu, x_logvar):
    x_logvar=torch.clamp(x_logvar, min= -10, max =10)
    return 0.5 * (np.log(2. * np.pi) + x_logvar + torch.exp(-x_logvar) * (x - x_mu)**2)


def kl_balancer(kl_all):
    kl_coeff_i = kl_all.mean(dim=0, keepdim=True) + 0.01
    kl_coeff_i /= kl_coeff_i.mean(dim=1)
    return kl_all * kl_coeff_i


def get_kl(delta_mu, delta_logvar, mu, logvar):
    return 0.5 * (delta_mu**2 * torch.exp(-logvar) + torch.exp(delta_logvar) - delta_logvar - 1).sum(1)


def kl_balancer(kl_all):
    kl_coeff_i = kl_all.mean(dim=0, keepdim=True) + 0.01
    kl_coeff_i /= kl_coeff_i.mean(dim=1)
    return kl_all * kl_coeff_i.detach()

