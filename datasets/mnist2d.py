import os
import numpy as np
from torch.utils.data import Dataset


class MNIST2D(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        self.shapes, self.labels, self.contexts = self.get_data()


    def get_data(self):
        if self.train:
            split_dir = os.path.join(self.data_dir, 'MNIST2D_test.xyz')
        else:
            split_dir = os.path.join(self.data_dir, 'MNIST2D_train.xyz')

        with open(split_dir, 'r') as infile:
            lines = infile.readlines()
            n_clouds = int(lines[0])
            
            i = 2
            labels = []
            shapes = []
            contexts = []
            while i < len(lines):
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

                shapes.append(points)
                contexts.append(context)
                i += n_points + 1
            
            assert len(shapes) == n_clouds
            assert len(labels) == n_clouds
            assert len(contexts) == n_clouds
        
        # print(shapes[:10], labels[:10], contexts[:10])
        return shapes, labels, contexts
            

    def __getitem__(self, idx):
        shapes = self.shapes[idx]
        if self.transform:
            shapes = self.transform(shapes)
        return shapes, self.labels[idx], self.contexts[idx]


    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = MNIST2D('/pio/scratch/1/mstyp/joint-clouds/data', train=False)
    for shape, label, context in dataset:
        print(len(shape), label, len(context))
        break