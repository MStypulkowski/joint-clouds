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

from models import DeepVAE
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
    data_train = MNIST2D(args.data_dir, n_points_per_cloud=args.n_points_per_cloud)
    print(f'Loaded train split with {len(data_train)} samples.')
    data_test = MNIST2D(args.data_dir, split='test', n_points_per_cloud=args.n_points_per_cloud)
    print(f'Loaded test split with {len(data_test)} samples.')
    dataloader_train = DataLoader(data_train, batch_size=args.bsz, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.bsz, shuffle=True)

    print('=' * 100)
    print('Preparing model...')

    model = DeepVAE(args.n_latents, data_train.x_dim, args.h_dim, args.hid_dim, args.e_dim, args.ze_dim, args.z_dim, args.n_points_per_cloud,
                    use_positional_encoding=args.use_positional_encoding, L=args.L).to(device)
    
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
    with wandb.init(project="joint-clouds", entity="joint-clouds", config=args, name=f'deep-VAE p(x) {datetime.now()}'):
        for epoch in range(args.n_epochs):
            pbar = tqdm(dataloader_train, desc=f'Epoch: {epoch}')
            for i, (x, y, _) in enumerate(pbar):
                if epoch == 0 and i == 0:
                    sample = x[0].numpy()
                    # title = f'data'
                    # plt.figure(figsize=(5, 10))
                    # plt.scatter(sample[:, 0], sample[:, 1])
                    # plt.title(title)
                    # plt.savefig(os.path.join(args.save_dir, 'figures', title + '.png'))
                    # plt.close()
                    sample = [[x1, x2] for (x1, x2) in sample]
                    table = wandb.Table(data=sample, columns=['x1', 'x2'])
                    wandb.log({f'Data': wandb.plot.scatter(table, 'x1', 'x2', title=f'Data')})

                optimizer.zero_grad()
                x = x.float().to(device)
                # data_mean, data_std = x.mean([0, 1]), x.std([0, 1])
                # x = (x - data_mean) / data_std
                # x += torch.rand_like(x) * 1e-2
                data_mean = x.mean([0, 1])
                x -= data_mean
                data_range = (x.reshape(-1, x.shape[-1]).max(0)[0] - x.reshape(-1, x.shape[-1]).min(0)[0])
                x /= data_range

                elbo, nll, kl_ze, kls, x_recon = model(x)

                if args.n_gpus > 1:
                    elbo = elbo.mean()
                    nll = nll.mean()
                    kl_ze = kl_ze.mean()
                    for i in kls:
                        wandb.log({i: kls[i].mean().item()})
                        kls[i] = f'{kls[i].mean().item():.4f}'

                nll *= args.beta
                if epoch < args.n_warmup_epochs:
                    nll.backward()
                else:
                    elbo.backward()
                optimizer.step()

                log_dict = OrderedDict(
                    {
                        'ELBO': '%.4f' % elbo.item(),
                        'NLL': '%.4f' % (nll.item() / args.beta),
                        'KL_ze': '%.4f' % kl_ze.item()
                    }
                )
                log_dict.update(kls)
                pbar.set_postfix(log_dict)

                wandb.log({
                    'ELBO': elbo.item(),
                    'NLL': (nll.item() / args.beta),
                    'KL_ze': kl_ze.item()
                })
            
            # evaluation
            if epoch % args.eval_frequency == 0:
                with torch.no_grad():
                    model.eval()

                    # generation
                    if args.n_gpus > 1:
                        samples = model.module.sample(args.n_samples, args.n_points_per_cloud_gen)
                    else:
                        samples = model.sample(args.n_samples, args.n_points_per_cloud_gen)
                    
                    # samples = samples * data_std + data_mean
                    samples = samples * data_range + data_mean

                    for i, sample in enumerate(samples):
                        sample = sample.cpu().numpy()
                        title = f'Epoch {epoch} sample {i}'
                        # plt.figure(figsize=(5, 10))
                        # plt.scatter(sample[:, 0], sample[:, 1], alpha=0.2)
                        # plt.title(title)
                        # plt.savefig(os.path.join(args.save_dir, 'figures', title + '.png'))
                        # plt.close()
                        sample = [[x1, x2] for (x1, x2) in sample]
                        table = wandb.Table(data=sample, columns=['x1', 'x2'])
                        wandb.log({title: wandb.plot.scatter(table, 'x1', 'x2', title=title)})
                    
                    # for i, _z in enumerate(zs):
                    #     for j, sample in enumerate(_z):
                    #         title = f'Sample {j} Z_{i}'
                    #         plt.figure(figsize=(5, 10))
                    #         plt.scatter(sample[:, 0], sample[:, 1], alpha=0.2)
                    #         plt.title(title)
                    #         plt.savefig(os.path.join(args.save_dir, 'figures', title + 'png'))
                    #         plt.close()

                    # reconstruction
                    for i in range(args.n_samples):
                        _x = x[i].cpu().numpy()
                        _x_recon = x_recon[i].cpu().numpy()
                        title = f'Epoch {epoch} recon {i}'
                        # plt.figure(figsize=(10, 10))
                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].scatter(_x[:, 0], _x[:, 1])
                        # ax[1].scatter(_x_recon[:, 0], _x_recon[:, 1])
                        # fig.suptitle(title)
                        # plt.savefig(os.path.join(args.save_dir, 'figures', title + '.png'))
                        # plt.close()
                        _x_recon = [[x1, x2] for (x1, x2) in _x_recon]
                        table = wandb.Table(data=_x_recon, columns=['x1', 'x2'])
                        wandb.log({title: wandb.plot.scatter(table, 'x1', 'x2', title=title)})

                        title = f'Epoch {epoch} recon {i} ref'
                        _x = [[x1, x2] for (x1, x2) in _x]
                        table = wandb.Table(data=_x, columns=['x1', 'x2'])
                        wandb.log({title: wandb.plot.scatter(table, 'x1', 'x2', title=title)})

                    # for i, _z in enumerate(zs_recon):
                    #     for j, sample in enumerate(_z[:args.n_samples]):
                    #         title = f'Recon {j} Z_{i}'
                    #         sample = sample.cpu().numpy()
                    #         plt.figure(figsize=(5, 10))
                    #         plt.scatter(sample[:, 0], sample[:, 1])
                    #         plt.title(title)
                    #         plt.savefig(os.path.join(args.save_dir, 'figures', title + 'png'))
                    #         plt.close()

                    model.train()


if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 3721))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    main()