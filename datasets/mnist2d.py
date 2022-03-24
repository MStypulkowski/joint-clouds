import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MNIST2D(Dataset):
    def __init__(self, data_dir, split='train', transform=ToTensor(), n_points_per_cloud=128):
        assert transform is not None # needs to contain at least ToTensor()

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.n_points_per_cloud = n_points_per_cloud

        self.clouds, self.labels, self.contexts = self.get_data()
        self.n_classes = len(np.unique(self.labels))
        self.x_dim = 2 # dimension of each point


    def get_data(self):
        if self.split == 'train':
            split_dir = os.path.join(self.data_dir, 'MNIST2D_train.xyz')
        else:
            split_dir = os.path.join(self.data_dir, 'MNIST2D_test.xyz')

        print(f'Loading MNIST2D {self.split} split from {split_dir}...')

        with open(split_dir, 'r') as infile:
            lines = infile.readlines()
            n_clouds = int(lines[0])
            
            i = 2
            labels = []
            clouds = []
            contexts = []
            while i < len(lines):
                if int(lines[i]) != 2:
                    i += int(lines[i + 1]) + 3
                    continue
                labels.append(int(lines[i]))
                i += 1
                n_points = int(lines[i])
                i += 1
                
                points = []
                context = []
                for j in range(n_points):
                    values = lines[i + j].split(' ')
                    points.append([float(values[0]), float(values[1])])
                    context.append(int(values[2]))
                
                assert len(points) == n_points
                assert len(context) == n_points

                clouds.append(np.array(points))
                contexts.append(np.array(context))
                i += n_points + 1
            
            # assert len(clouds) == n_clouds
            # assert len(labels) == n_clouds
            # assert len(contexts) == n_clouds
            assert len(clouds) == len(labels) == len(contexts)
        
        return clouds, labels, contexts
        
    def __getitem__(self, idx):
        cloud = self.transform(self.clouds[idx]).squeeze(0)
        context = torch.tensor(self.contexts[idx])

        n_points = cloud.shape[0]

        if n_points < self.n_points_per_cloud:
            # resample points if too few and add some noise
            weights = torch.ones(n_points)
            point_ids = torch.multinomial(weights, self.n_points_per_cloud, replacement=True)
            cloud = cloud[point_ids, :]
            cloud += torch.rand(cloud.shape) / 10
            context = context[point_ids]

        elif n_points > self.n_points_per_cloud:
            # randomly pick a subset of the cloud if too many points
            permutation = torch.randperm(n_points)
            cloud = cloud[permutation, :]
            cloud = cloud[:self.n_points_per_cloud, :]
            context = context[permutation]
            context = context[:self.n_points_per_cloud]
        
        return cloud.squeeze(0), self.labels[idx], context

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = MNIST2D('/pio/scratch/1/mstyp/joint-clouds/data', transform=ToTensor(), split='test')

    print(dataset.n_classes, dataset.x_dim)

    for i, (cloud, label, context) in enumerate(dataset):
        print(cloud.shape, label, len(context))

        if i == 10:
            break