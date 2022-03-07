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

from models import TopDownVAE

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
    data_train = MNIST2D(args.data_dir)
    data_test = MNIST2D(args.data_dir, split='test')

    dataloader_train = DataLoader(data_train, batch_size=args.bsz, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.bsz, shuffle=True)

    print('=' * 100)
    print('Preparing model...')

    model = TopDownVAE(data_train.x_dim, data_train.n_classes,
                            h_dim=args.h_dim, e_dim=args.e_dim, z1_dim=args.z1_dim, z2_dim=args.z2_dim, 
                            n_layers=args.n_layers).to(device)
    
    print(f'Number of trainable parameters: {count_trainable_parameters(model)}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    classification_loss = nn.CrossEntropyLoss()

    print('=' * 100)
    print('Training...')
    with wandb.init(project="joint-clouds", entity="joint-clouds", config=args, name=f'top-down-VAE p(x)(y|x) {datetime.now()}'):
        for epoch in range(args.n_epochs):
            pbar = tqdm(dataloader_train, desc=f'Epoch: {epoch}')
            for x, y, _ in pbar:
                optimizer.zero_grad()
                x = x.float().to(device)
                y = y.to(device)

                elbo, logits = model(x)
                class_loss_val = classification_loss(logits, y)
                accuracy = (logits.argmax(1) == y).float().mean() * 100

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

                wandb.log({
                    'Total loss': loss.item(),
                    'ELBO': elbo.item(),
                    'Class. loss': class_loss_val.item(),
                    'Class. accuracy': accuracy,
                })
            
            # evaluation
            if epoch % args.eval_frequency == 0:
                with torch.no_grad():
                    model.eval()

                    # generation
                    samples = model.sample(args.n_samples, args.n_points_per_cloud)
                    for i, sample in enumerate(samples):
                        sample = sample.cpu().numpy().T
                        sample = [[x1, x2] for (x1, x2) in sample]
                        table = wandb.Table(data=sample, columns=['x1', 'x2'])
                        wandb.log({f'Sample {i} epoch {epoch}': wandb.plot.scatter(table, 'x1', 'x2', title=f'Sample {i} epoch {epoch}')})
                    
                    # ELBO + classification
                    test_shape_count = 0
                    test_elbo_acc = 0.
                    test_class_loss_val_acc = 0.
                    test_accuracy_acc = 0.

                    for x, y, _ in tqdm(dataloader_test, desc=f'Evaluation'):
                        x = x.float().to(device)
                        y = y.to(device)

                        test_elbo, test_logits = model(x)
                        test_class_loss_val = classification_loss(test_logits, y)
                        test_accuracy = (test_logits.argmax(1) == y).float().mean() * 100

                        loss = elbo + class_loss_val

                        test_shape_count += len(x)
                        test_elbo_acc += test_elbo.item() * len(x)
                        test_class_loss_val_acc += test_class_loss_val * len(x)
                        test_accuracy_acc += test_accuracy * len(x)

                    test_loss_final = (test_elbo_acc + test_class_loss_val_acc) / test_shape_count
                    test_elbo_final = test_elbo_acc / test_shape_count
                    test_class_loss_val_final = test_class_loss_val_acc / test_shape_count
                    test_accuracy_final = test_accuracy_acc / test_shape_count

                    print(f'Test loss: {test_loss_final:.4f} Test ELBO: {test_elbo_final:.4f} Test class. loss: {test_class_loss_val_final:.4f} Test class. accuracy: {test_accuracy_final:.2f}')
                    
                    wandb.log({
                        'Epoch': epoch,
                        'Test loss': test_loss_final,
                        'Test ELBO': test_elbo_final,
                        'Test class. loss': test_class_loss_val_final,
                        'Test class. accuracy': test_accuracy_final,
                    })

                    model.train()


if __name__ == '__main__':
    main()