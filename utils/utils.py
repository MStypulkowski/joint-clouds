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


def get_kl(delta_mu, delta_logvar, mu, logvar):
    return 0.5 * (delta_mu**2 * torch.exp(-logvar) + torch.exp(delta_logvar) - delta_logvar - 1).sum()