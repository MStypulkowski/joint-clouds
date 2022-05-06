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

from models import ConditionalTopDownVAE

from utils import count_trainable_parameters, kl_balancer

import wandb
wandb.login()


@hydra.main(config_path='./configs', config_name='config7')
def main(args):
    # experiment = Experiment(project_name='joint-clouds')
    # experiment.log_parameters(args)
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

    model = ConditionalTopDownVAE(data_train.x_dim, data_train.n_classes,
                                h_dim=args.h_dim, e_dim=args.e_dim, ze_dim=args.ze_dim, z_dim=args.z_dim, n_latents=args.n_latents,
                                encoder_hid_dim=args.encoder_hid_dim, decoder_hid_dim=args.decoder_hid_dim,
                                encoder_n_resnet_blocks=args.encoder_n_resnet_blocks, decoder_n_resnet_blocks=args.decoder_n_resnet_blocks,
                                activation=args.activation, last_activation=args.last_activation, use_batchnorms=args.use_batchnorms, 
                                use_lipschitz_norm=args.use_lipschitz_norm, lipschitz_loss_weight=args.lipschitz_loss_weight, 
                                use_positional_encoding=args.use_positional_encoding, L=args.L).to(device)
    
    if args.n_gpus > 1:
        model = torch.nn.DataParallel(model)

    print(model)
    print(f'Number of trainable parameters: {count_trainable_parameters(model)}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    classification_loss = nn.CrossEntropyLoss()

    print('=' * 100)
    print('Training...')
    with wandb.init(project="joint-clouds", entity="joint-clouds", config=args, name=f'conditional-top-down-VAE p(x) {datetime.now()}', settings=wandb.Settings(start_method="fork")):
    # while True:
    # with experiment.train():
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
                # x = x.permute(0, 2, 1)
                # x += torch.rand_like(x) * 1e-2
                # y = y.to(device)
                data_mean = x.mean([0, 1])
                x -= data_mean
                data_range = (x.reshape(-1, x.shape[-1]).max(0)[0] - x.reshape(-1, x.shape[-1]).min(0)[0])
                x /= data_range
                x += torch.rand_like(x) * 1e-2

                # elbo, logits, nll, kl_z1, kl_z2 = model(x)
                if epoch % args.eval_frequency == 0 and i == len(dataloader_train) - 1:
                    elbo, nll, kl_ze, kls, x_recon, zs_recon = model(x, epoch=epoch, save_dir=args.save_dir)
                else:
                    elbo, nll, kl_ze, kls = model(x)

                # class_loss_val = classification_loss(logits, y)
                # accuracy = (logits.argmax(1) == y).float().mean() * 100

                # loss = elbo + class_loss_val
                # loss.backward()
                if args.n_gpus > 1:
                    elbo = elbo.mean()
                    nll = nll.mean()
                    kl_ze = kl_ze.mean()
                    for i in kls:
                        wandb.log({i: kls[i].mean().item()})
                        kls[i] = '%.4f' % kls[i].mean().item()

                if args.use_lipschitz_norm:
                    lipschitz_loss = model.lipschitz_loss() if args.n_gpus == 1 else model.module.lipschitz_loss()
                else:
                    lipschitz_loss = 0.

                if epoch < args.n_warmup_epochs:
                    (nll + lipschitz_loss).backward()
                else:
                    (elbo + lipschitz_loss).backward()
                optimizer.step()

                if args.use_lipschitz_norm:
                    log_dict = OrderedDict(
                        {
                            # 'Total loss': '%.4f' % loss.item(),
                            'ELBO': '%.4f' % elbo.item(),
                            'NLL': '%.4f' % nll.item(),
                            'KL_ze': '%.4f' % kl_ze.item(),
                            'lipschitz_loss': '%.0f' % lipschitz_loss.item(),
                            # 'Class. loss': '%.4f' % class_loss_val.item(),
                            # 'Class. accuracy': '%.2f' % accuracy,
                        }
                    )
                else:
                    log_dict = OrderedDict(
                        {
                            # 'Total loss': '%.4f' % loss.item(),
                            'ELBO': '%.4f' % elbo.item(),
                            'NLL': '%.4f' % nll.item(),
                            'KL_ze': '%.4f' % kl_ze.item(),
                            # 'Class. loss': '%.4f' % class_loss_val.item(),
                            # 'Class. accuracy': '%.2f' % accuracy,
                        }
                    )
                log_dict.update(kls)
                pbar.set_postfix(log_dict)
                
                wandb.log({
                    'ELBO': elbo.item(),
                    'NLL': nll.item(),
                    'KL_ze': kl_ze.item()
                })
            
            # evaluation
            if epoch % args.eval_frequency == 0:
                with torch.no_grad():
                    model.eval()

                    # generation
                    if args.n_gpus > 1:
                        samples, zs = model.module.sample(args.n_samples, args.n_points_per_cloud_gen)
                    else:
                        samples, zs = model.sample(args.n_samples, args.n_points_per_cloud_gen)
                    
                    samples = samples * data_range + data_mean
                    for i, sample in enumerate(samples):
                        sample = sample.cpu().numpy()
                        # title = f'Epoch {epoch} sample {i}'
                        # plt.figure(figsize=(5, 10))
                        # plt.scatter(sample[:, 0], sample[:, 1], alpha=0.2)
                        # plt.title(title)
                        # plt.savefig(os.path.join(args.save_dir, 'figures', title + '.png'))
                        # plt.close()
                        sample = [[x1, x2] for (x1, x2) in sample]
                        table = wandb.Table(data=sample, columns=['x1', 'x2'])
                        wandb.log({f'Sample {i} epoch {epoch}': wandb.plot.scatter(table, 'x1', 'x2', title=f'Sample {i} epoch {epoch}')})
                    
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
                        # title = f'Epoch {epoch} recon {i}'
                        # plt.figure(figsize=(10, 10))
                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].scatter(_x[:, 0], _x[:, 1])
                        # ax[1].scatter(_x_recon[:, 0], _x_recon[:, 1])
                        # fig.suptitle(title)
                        # plt.savefig(os.path.join(args.save_dir, 'figures', title + '.png'))
                        # plt.close()
                        name = f'Recon {i} epoch {epoch} ref'
                        sample = [[x1, x2] for (x1, x2) in _x]
                        table = wandb.Table(data=sample, columns=['x1', 'x2'])
                        wandb.log({name: wandb.plot.scatter(table, 'x1', 'x2', title=name)})
                        name = f'Recon {i} epoch {epoch}'
                        sample = [[x1, x2] for (x1, x2) in _x_recon]
                        table = wandb.Table(data=sample, columns=['x1', 'x2'])
                        wandb.log({name: wandb.plot.scatter(table, 'x1', 'x2', title=name)})

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