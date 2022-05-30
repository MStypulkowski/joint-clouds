import os
import yaml
from datetime import datetime
from collections import OrderedDict
import hydra
# from tqdm import tqdm
import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from datasets import MNIST2D
from datasets.datasets_pointflow import (
    CIFDatasetDecorator,
    ShapeNet15kPointClouds,
    CIFDatasetDecoratorMultiObject,
)

from models import SingleShapeDeepVAE
from models.utils import count_trainable_parameters, plot_3d_cloud

import wandb
wandb.login()

@hydra.main(config_path='./configs', config_name='config')
def main(args):
    print(args)

    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print('=' * 100)
    print('Preparing dataset...')
    
    data_idx = 1

    if args.dataset == 'minst2d':
        data_sample = MNIST2D(args.data_dir, n_points_per_cloud=args.n_points_per_cloud)
        # data_test = MNIST2D(args.data_dir, split='test', n_points_per_cloud=args.n_points_per_cloud)
        x_dim = 2

    elif args.dataset == 'shapenet':
        data_sample = ShapeNet15kPointClouds(root_dir=args.data_dir, categories=args.classes, 
                                            tr_sample_size=args.n_points_per_cloud, te_sample_size=0, split='train', 
                                            normalize_per_shape=False, normalize_std_per_axis=False, 
                                            random_subsample=False, all_points_mean=None, all_points_std=None)[data_idx]['train_points']
        x_dim = 3

    # print(f'Loaded train split with {len(data_train)} samples.')
    # print(f'Loaded test split with {len(data_test)} samples.')
    # dataloader_train = DataLoader(data_train, batch_size=args.bsz, shuffle=True)
    # dataloader_test = DataLoader(data_test, batch_size=args.bsz, shuffle=True)
    print(f'Loaded shape number {data_idx}.')

    plot_3d_cloud(data_sample, args.log_dir, 'data_sample' + str(data_idx))
    data_sample = data_sample.float().to(device)

    print('=' * 100)
    print('Preparing model...')

    model = SingleShapeDeepVAE(args.n_latents, x_dim, args.h_dim, args.hid_dim, args.z_dim, args.n_layers).to(device)

    print(model)
    print(f'Number of trainable parameters: {count_trainable_parameters(model)}')

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print('=' * 100)
    print('Training...')

    # with wandb.init(project="joint-clouds", entity="joint-clouds", config=args, name=f'single-VAE p(x) {datetime.now()}'):
    pbar = tqdm.trange(args.n_epochs)
    for epoch in pbar:
        x = data_sample + torch.rand_like(data_sample) * 1e-2
        x = x.unsqueeze(0)

        elbo, nll, kl_z_dict, x_recon = model(x)

        optimizer.zero_grad()
        if epoch < args.n_warmup_epochs:
            nll.backward()
        else:
            elbo.backward()
        optimizer.step()

        if epoch % 100 == 0:
            output_str = f'Epoch {epoch} ELBO {elbo.item():.4f} NLL {nll.item():.4f} '

            for kl_i in kl_z_dict:
                output_str += f'{kl_i} {kl_z_dict[kl_i].item():.4f} '
            
            tqdm.tqdm.write(output_str)
        
        if epoch > 0 and epoch % 1000 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.9

            with torch.no_grad():
                model.eval()
                sample = model.sample(1, args.n_points_per_cloud_gen, device=device).cpu().squeeze(0)
                plot_3d_cloud(sample, args.log_dir, f'sample_{epoch}')


if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 3721))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    main()