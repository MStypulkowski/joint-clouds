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