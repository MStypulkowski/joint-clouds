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

from models import TopDownVAE, ConditionalTopDownVAE

from utils import count_trainable_parameters

# import wandb
# wandb.login()


@hydra.main(config_path='./configs', config_name='config')
def main(args):
    # experiment = Experiment(project_name='joint-clouds')
    # experiment.log_parameters(args)

    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print('=' * 100)
    print('Preparing dataset...')
    data_train = MNIST2D(args.data_dir)
    print(f'Loaded train split with {len(data_train)} samples.')
    data_test = MNIST2D(args.data_dir, split='test')
    print(f'Loaded test split with {len(data_test)} samples.')
    dataloader_train = DataLoader(data_train, batch_size=args.bsz, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.bsz, shuffle=True)

    print('=' * 100)
    print('Preparing model...')

    model = ConditionalTopDownVAE(data_train.x_dim, data_train.n_classes,
                            h1_dim=args.h1_dim, h2_dim=args.h2_dim, e_dim=args.e_dim, ze_dim=args.ze_dim, z1_dim=args.z1_dim, z2_dim=args.z2_dim, 
                            hid_dim=args.hid_dim, n_layers=args.n_layers, activation=args.activation).to(device)
    
    if args.n_gpus > 1:
        model = torch.nn.DataParallel(model)

    print(model)
    print(f'Number of trainable parameters: {count_trainable_parameters(model)}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    classification_loss = nn.CrossEntropyLoss()

    print('=' * 100)
    print('Training...')
    # with wandb.init(project="joint-clouds", entity="joint-clouds", config=args, name=f'conditional-top-down-VAE p(x) {datetime.now()}', settings=wandb.Settings(start_method="fork")):
    # with experiment.train():
    for epoch in range(args.n_epochs):
        pbar = tqdm(dataloader_train, desc=f'Epoch: {epoch}')
        for i, (x, y, _) in enumerate(pbar):
            if epoch == 0 and i == 0:
                sample = x[0].numpy()
                title = f'data'
                plt.figure(figsize=(5, 10))
                plt.scatter(sample[:, 0], sample[:, 1])
                plt.title(title)
                plt.savefig(os.path.join(args.save_dir, 'figures', title + '.png'))
                plt.close()
                # sample = [[x1, x2] for (x1, x2) in sample]
                # table = wandb.Table(data=sample, columns=['x1', 'x2'])
                # wandb.log({f'Data': wandb.plot.scatter(table, 'x1', 'x2', title=f'Data')})

            optimizer.zero_grad()
            x = x.float().to(device)
            data_mean, data_std = x.mean([0, 1]), x.std([0, 1])
            x = (x - data_mean) / data_std
            x = x.permute(0, 2, 1)
            # y = y.to(device)

            # elbo, logits, nll, kl_z1, kl_z2 = model(x)
            if i == 0:
                elbo, nll, kl_z1, kl_z2, kl_ze, x_recon = model(x, epoch=epoch)
            else:
                elbo, nll, kl_z1, kl_z2, kl_ze, x_recon = model(x)
            # class_loss_val = classification_loss(logits, y)
            # accuracy = (logits.argmax(1) == y).float().mean() * 100

            # loss = elbo + class_loss_val
            # loss.backward()
            if args.n_gpus > 1:
                elbo = elbo.mean()
                nll = nll.mean()
                kl_z1 = kl_z1.mean()
                kl_z2 = kl_z2.mean()
                kl_ze = kl_ze.mean()
            if epoch < args.n_warmup_epochs:
                nll.backward()
            else:
                elbo.backward()
            optimizer.step()

            pbar.set_postfix(OrderedDict(
                {
                    # 'Total loss': '%.4f' % loss.item(),
                    'ELBO': '%.4f' % elbo.item(),
                    'NLL': '%.4f' % nll.item(),
                    'KL_z1': '%.4f' % kl_z1.item(),
                    'KL_z2': '%.4f' % kl_z2.item(),
                    'KL_ze': '%.4f' % kl_ze.item(),
                    # 'Class. loss': '%.4f' % class_loss_val.item(),
                    # 'Class. accuracy': '%.2f' % accuracy,
                }
            ))

            # wandb.log({
            #     # 'Total loss': loss.item(),
            #     'ELBO': elbo.item(),
            #     'NLL': nll.item(),
            #     'KL_z1': kl_z1.item(),
            #     'KL_z2': kl_z2.item(),
            #     'KL_ze': kl_ze.item(),
            #     # 'Class. loss': class_loss_val.item(),
            #     # 'Class. accuracy': accuracy,
            # })

            # experiment.log_metric({
            #     # 'Total loss': loss.item(),
            #     'ELBO': elbo.item(),
            #     'NLL': nll.item(),
            #     'KL_z1': kl_z1.item(),
            #     'KL_z2': kl_z2.item(),
            #     'KL_ze': kl_ze.item(),
            #     # 'Class. loss': class_loss_val.item(),
            #     # 'Class. accuracy': accuracy,
            # })
        
        # evaluation
        if epoch % args.eval_frequency == 0:
            with torch.no_grad():
                model.eval()

                # generation
                if args.n_gpus > 1:
                    samples = model.module.sample(args.n_samples, args.n_points_per_cloud_gen)
                else:
                    samples = model.sample(args.n_samples, args.n_points_per_cloud_gen)
                samples = samples * data_std + data_mean
                for i, sample in enumerate(samples):
                    sample = sample.cpu().numpy()
                    title = f'Epoch {epoch} sample {i}'
                    plt.figure(figsize=(5, 10))
                    plt.scatter(sample[:, 0], sample[:, 1])
                    plt.title(title)
                    plt.savefig(os.path.join(args.save_dir, 'figures', title + '.png'))
                    plt.close()
                    # sample = [[x1, x2] for (x1, x2) in sample]
                    # table = wandb.Table(data=sample, columns=['x1', 'x2'])
                    # wandb.log({f'Sample {i} epoch {epoch}': wandb.plot.scatter(table, 'x1', 'x2', title=f'Sample {i} epoch {epoch}')})
                
                # reconstruction
                for i in range(args.n_samples):
                    _x = x[i].cpu().numpy().T
                    _x_recon = x_recon[i].cpu().numpy().T
                    title = f'Epoch {epoch} recon {i}'
                    plt.figure(figsize=(10, 10))
                    fig, ax = plt.subplots(1, 2)
                    ax[0].scatter(_x[:, 0], _x[:, 1])
                    ax[1].scatter(_x_recon[:, 0], _x_recon[:, 1])
                    fig.suptitle(title)
                    plt.savefig(os.path.join(args.save_dir, 'figures', title + '.png'))
                    plt.close()
                # # ELBO + classification
                # test_shape_count = 0
                # test_elbo_acc = 0.
                # test_nll_acc = 0.
                # test_kl_z1_acc = 0.
                # test_kl_z2_acc = 0.
                # test_class_loss_val_acc = 0.
                # test_accuracy_acc = 0.

                # for x, y, _ in tqdm(dataloader_test, desc=f'Evaluation'):
                #     x = x.float().to(device)
                #     y = y.to(device)

                #     test_elbo, test_logits, test_nll, test_kl_z1, test_kl_z2 = model(x)
                #     test_class_loss_val = classification_loss(test_logits, y)
                #     test_accuracy = (test_logits.argmax(1) == y).float().mean() * 100

                #     loss = elbo + class_loss_val

                #     test_shape_count += len(x)
                #     test_elbo_acc += test_elbo.item() * len(x)
                #     test_nll_acc += test_nll.item() * len(x)
                #     test_kl_z1_acc += test_kl_z1.item() * len(x)
                #     test_kl_z2_acc += test_kl_z2.item() * len(x)
                #     test_class_loss_val_acc += test_class_loss_val * len(x)
                #     test_accuracy_acc += test_accuracy * len(x)

                # test_loss_final = (test_elbo_acc + test_class_loss_val_acc) / test_shape_count
                # test_elbo_final = test_elbo_acc / test_shape_count
                # test_nll_final = test_nll_acc / test_shape_count
                # test_kl_z1_final = test_kl_z1_acc / test_shape_count
                # test_kl_z2_final = test_kl_z2_acc / test_shape_count
                # test_class_loss_val_final = test_class_loss_val_acc / test_shape_count
                # test_accuracy_final = test_accuracy_acc / test_shape_count

                # print(f'Test loss: {test_loss_final:.4f} Test ELBO: {test_elbo_final:.4f} Test NLL: {test_nll_final:.4f} Test KL_z1: {test_kl_z1_final:.4f} Test KL_z2: {test_kl_z2_final:.4f} Test class. loss: {test_class_loss_val_final:.4f} Test class. accuracy: {test_accuracy_final:.2f}')
                
                # wandb.log({
                #     'Epoch': epoch,
                #     'Test loss': test_loss_final,
                #     'Test ELBO': test_elbo_final,
                #     'Test NLL': test_nll_final,
                #     'Test KL_z1': test_kl_z1_final,
                #     'Test KL_z2': test_kl_z2_final,
                #     'Test class. loss': test_class_loss_val_final,
                #     'Test class. accuracy': test_accuracy_final,
                # })

                model.train()


if __name__ == '__main__':
    main()