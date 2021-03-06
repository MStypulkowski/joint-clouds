import os
import yaml
from datetime import datetime
from collections import OrderedDict
import hydra
from tqdm import tqdm

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

from models import SimpleVAE
from models.utils import count_trainable_parameters

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
    
    if args.dataset == 'minst2d':
        data_train = MNIST2D(args.data_dir, n_points_per_cloud=args.n_points_per_cloud)
        # data_test = MNIST2D(args.data_dir, split='test', n_points_per_cloud=args.n_points_per_cloud)

    elif args.dataset == 'shapenet':
        data_train = ShapeNet15kPointClouds(root_dir=args.data_dir, categories=args.classes, 
                                            tr_sample_size=args.n_points_per_cloud, te_sample_size=0, split='train', 
                                            normalize_per_shape=False, normalize_std_per_axis=False, 
                                            random_subsample=False, all_points_mean=None, all_points_std=None)

    print(f'Loaded train split with {len(data_train)} samples.')
    # print(f'Loaded test split with {len(data_test)} samples.')
    dataloader_train = DataLoader(data_train, batch_size=args.bsz, shuffle=True)
    # dataloader_test = DataLoader(data_test, batch_size=args.bsz, shuffle=True)
    
    print('=' * 100)
    print('Preparing model...')

    model = SimpleVAE(data_train.x_dim, args.h_dim, args.z_dim, args.emb_dim,
                    args.encoder_hid_dim, args.encoder_n_layers, args.decoder_hid_dim, 
                    args.decoder_n_layers, use_hypernet=args.use_hypernet, hyper_hid_dim=args.hyper_hid_dim, hyper_n_layers=args.hyper_n_layers).to(device)
    
    if args.n_gpus > 1:
        model = torch.nn.DataParallel(model)

    print(model)
    print(f'Number of trainable parameters: {count_trainable_parameters(model)}')

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print('=' * 100)
    print('Training...')
    with wandb.init(project="joint-clouds", entity="joint-clouds", config=args, name=f'simple-VAE p(x) {datetime.now()}'):
        for epoch in range(args.n_epochs):
            pbar = tqdm(dataloader_train, desc=f'Epoch: {epoch}')
            for i, (x, y, _) in enumerate(pbar):
                if epoch == 0 and i == 0:
                    sample = x[0].numpy()
                    sample = [[x1, x2] for (x1, x2) in sample]
                    table = wandb.Table(data=sample, columns=['x1', 'x2'])
                    wandb.log({f'Data': wandb.plot.scatter(table, 'x1', 'x2', title=f'Data')})

                optimizer.zero_grad()
                x = x.float().to(device)
                data_mean, data_std = x.mean([0, 1]), x.std([0, 1])
                x = (x - data_mean) / data_std
                x += torch.rand_like(x) * 1e-2
                # data_mean = x.mean([0, 1])
                # x -= data_mean
                # data_range = (x.reshape(-1, x.shape[-1]).max(0)[0] - x.reshape(-1, x.shape[-1]).min(0)[0])
                # x /= data_range
                # x += torch.rand_like(x) * 1e-2

                nll, kl_z, kl_ze, x_recon = model(x)

                if args.n_gpus > 1:
                    nll = nll.mean()
                    kl_ze = kl_ze.mean()
                    kl_z = kl_z.mean()
                elbo = nll + kl_ze + kl_z

                if epoch < args.n_warmup_epochs:
                    nll.backward()
                else:
                    elbo.backward()
                optimizer.step()

                log_dict = OrderedDict(
                    {
                        'ELBO': f'{elbo.item():.4f}',
                        'NLL': f'{nll.item():.4f}',
                        'KL_ze': f'{kl_ze.item():.4f}',
                        'KL_z': f'{kl_z.item():.4f}'
                    }
                )
                pbar.set_postfix(log_dict)

                wandb.log({
                    'ELBO': elbo.item(),
                    'NLL': nll.item(),
                    'KL_ze': kl_ze.item(),
                    'KL_z': kl_z.item()
                })
            
            # evaluation
            if epoch % args.eval_frequency == 0:
                with torch.no_grad():
                    model.eval()

                    # generation
                    if args.n_gpus > 1:
                        samples = model.module.sample(args.n_samples, args.n_points_per_cloud_gen, data_train.x_dim)
                    else:
                        samples = model.sample(args.n_samples, args.n_points_per_cloud_gen, data_train.x_dim)
                    
                    samples = samples * data_std + data_mean
                    # samples = samples * data_range + data_mean

                    for i, sample in enumerate(samples):
                        sample = sample.cpu().numpy()
                        title = f'Epoch {epoch} sample {i}'
                        sample = [[x1, x2] for (x1, x2) in sample]
                        table = wandb.Table(data=sample, columns=['x1', 'x2'])
                        wandb.log({title: wandb.plot.scatter(table, 'x1', 'x2', title=title)})

                    # reconstruction
                    for i in range(args.n_samples):
                        _x = x[i].cpu().numpy()
                        _x_recon = x_recon[i].cpu().numpy()
                        title = f'Epoch {epoch} recon {i}'
                        _x_recon = [[x1, x2] for (x1, x2) in _x_recon]
                        table = wandb.Table(data=_x_recon, columns=['x1', 'x2'])
                        wandb.log({title: wandb.plot.scatter(table, 'x1', 'x2', title=title)})

                        title = f'Epoch {epoch} recon {i} ref'
                        _x = [[x1, x2] for (x1, x2) in _x]
                        table = wandb.Table(data=_x, columns=['x1', 'x2'])
                        wandb.log({title: wandb.plot.scatter(table, 'x1', 'x2', title=title)})

                    model.train()


if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 3721))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    main()