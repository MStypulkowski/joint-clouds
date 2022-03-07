import os
import hydra
import yaml
from tqdm import tqdm
from collections import OrderedDict

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from datasets import MNIST2D

from models import HierarchicalVAE

from utils import count_trainable_parameters

import wandb
wandb.login()


@hydra.main(config_path='./configs', config_name='config')
def main(args):
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print('=' * 100)
    print('Preparing dataset...')
    # data_train = MNIST2D(args.data_dir)
    data_test = MNIST2D(args.data_dir, split='test')

    # dataloader_train = DataLoader(data_train, batch_size=args.bsz, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.bsz, shuffle=True)

    print('=' * 100)
    print('Preparing model...')

    model = HierarchicalVAE(data_test.x_dim, data_test.n_classes,
                            h_dim=args.h_dim, e_dim=args.e_dim, z1_dim=args.z1_dim, z2_dim=args.z2_dim).to(device)
    
    print(f'Number of trainable parameters: {count_trainable_parameters(model)}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    classification_loss = nn.CrossEntropyLoss

    print('=' * 100)
    print('Training...')
    # # with wandb.init(project="joint-clouds", entity="joint-clouds", config=args, name='H-VAE p(x) * p(y|x)'):
    for epoch in range(args.n_epochs):
        pbar = tqdm(dataloader_test, desc=f'Epoch: {epoch}')
        for x, y, _ in pbar:
            optimizer.zero_grad()
            print(x.shape, y.shape)
            x = x.to(device)
            y = y.to(device)

            elbo, logits = model(x)
            class_loss_val = classification_loss(logits, y)
            print(logits.shape)
            accuracy = logits.max()

            loss = elbo + class_loss_val

            loss.backward()
            optimizer.step()

            pbar.set_postfix(OrderedDict(
                {
                    'Total loss': '%.4f' % loss.item(),
                    'ELBO': '%.4f' % elbo.item(),
                    'Class. loss': '%.4f' % class_loss_val.item(),
                    'Class. accuracy': '%.2f' % accuracy,
                }
            ))


if __name__ == '__main__':
    main()