import os
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from torch.utils import data
import random
import typing as t


# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15KPC(Dataset):
    def __init__(self, root_dir, root_embs_dir, subdirs, tr_sample_size=10000,
                 te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, random_subsample=False,
                 normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None,
                 input_dim=3):
        self.root_dir = root_dir
        self.root_embs_dir = root_embs_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim

        angles = (-np.pi/2, 0, np.pi)
        self.rotation_matrix = np.array(
            [[1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]]
        ) @ np.array(
            [[np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]]
        ) @ np.array(
            [[np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]]
        )

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)=
                except:
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)=

        # Rotate for better plotting view
        self.all_points = self.all_points @ self.rotation_matrix

        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))

        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'idx': idx,
            'train_points': tr_out,
            'test_points': te_out,
            'mean': m, 'std': s, 'cate_idx': cate_idx,
            'sid': sid, 'mid': mid
        }


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ShapeNetCore.v2.PC15k",
                 root_embs_dir="data/ShapeNetCore.v2.PC15k_embs",
                 categories=['chair'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root_dir = root_dir
        self.root_embs_dir = root_embs_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeNet15kPointClouds, self).__init__(
            root_dir, root_embs_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class CIFDatasetDecorator(Dataset):
    def __init__(self, dataset: Uniform15KPC):
        super().__init__()
        self.dataset = dataset
        self.instance_point_train_indices = np.stack(
            np.meshgrid(
                range(dataset.train_points.shape[0]),
                range(dataset.train_points.shape[1])
            ), axis=-1
        ).transpose((1, 0, 2)).reshape((-1, 2))

        self.instance_point_test_indices = np.stack(
            np.meshgrid(
                range(dataset.test_points.shape[0]),
                range(dataset.test_points.shape[1])
            ), axis=-1
        ).transpose((1, 0, 2))

        self.test_point_range = np.arange(dataset.test_points.shape[1])

    def __getitem__(self, idx: int) -> t.Dict[str, torch.Tensor]:
        # rewritting the original __getitem__ of dataset from PointFlow
        # to omit additional sampling of points inside an instance of a class

        # -> index to a particular instance
        corresponding_original_index = (
            self.instance_point_train_indices[idx][0]
        )
        train_point_coords = self.instance_point_train_indices[idx]

        available_test_points = self.instance_point_test_indices[
            corresponding_original_index
        ]

        if self.dataset.random_subsample:
            # does not matter which testing_point we will take from the testing
            test_point_coords = available_test_points[
                np.random.choice(self.test_point_range)
            ]
        else:
            test_point_coords = available_test_points[0]

        tr_out = self.dataset.train_points[
            train_point_coords[0],
            train_point_coords[1]
        ]
        tr_out = torch.from_numpy(tr_out).float()

        te_out = self.dataset.test_points[
            test_point_coords[0],
            test_point_coords[1]
        ]
        te_out = torch.from_numpy(te_out).float()

        m, s = self.dataset.get_pc_stats(corresponding_original_index)
        cate_idx = self.dataset.cate_idx_lst[corresponding_original_index]
        sid, mid = self.dataset.all_cate_mids[corresponding_original_index]
        ws = self.dataset.all_ws[corresponding_original_index]

        return {
            "idx": corresponding_original_index,
            "train_points": tr_out,
            "test_points": te_out,
            "mean": m,
            "std": s,
            "cate_idx": cate_idx,
            "sid": sid,
            "mid": mid,
            "w": ws
        }

    def __len__(self) -> int:
        return len(self.instance_point_train_indices)

    def renormalize(self, mean, std):
        self.dataset.renormalize(mean, std)

    def get_pc_stats(self, idx):
        return self.dataset.get_pc_stats(idx)

    @property
    def all_points_mean(self):
        return self.dataset.all_points_mean

    @property
    def all_points_std(self):
        return self.dataset.all_points_std

    @property
    def shuffle_idx(self):
        return self.dataset.shuffle_idx

    @property
    def all_points(self):
        return self.dataset.all_points

    @property
    def dataset_original_length(self):
        return len(self.dataset)


class CIFDatasetDecoratorMultiObject(Dataset):
    def __init__(
        self,
        dataset: Uniform15KPC,
        num_of_points_per_object: int
    ):
        super().__init__()
        self.dataset = dataset
        self.num_of_points_per_object = num_of_points_per_object

    def __getitem__(self, idx: int) -> t.Dict[str, torch.Tensor]:
        instance_index = (
            (idx * self.num_of_points_per_object)
            // self.dataset.train_points.shape[1]
        )
        data_dict = self.dataset[instance_index]

        start_index = (idx * self.num_of_points_per_object) % self.dataset.train_points.shape[1]
        end_index = min(
            start_index + self.num_of_points_per_object,
            self.dataset.train_points.shape[1]
        )

        # refinement to match dimensions for each tensor in the batch
        start_index = end_index - self.num_of_points_per_object

        indices = np.arange(start_index, end_index)
        np.random.shuffle(indices)

        points_to_decode = self.dataset.train_points[instance_index, indices]
        data_dict["points_to_decode"] = points_to_decode
        return data_dict

    def __len__(self) -> int:
        return (
            self.dataset.train_points.shape[0]
            * self.dataset.train_points.shape[1]
            // self.num_of_points_per_object
        )

    def renormalize(self, mean, std):
        self.dataset.renormalize(mean, std)

    def get_pc_stats(self, idx):
        return self.dataset.get_pc_stats(idx)

    @property
    def all_points_mean(self):
        return self.dataset.all_points_mean

    @property
    def all_points_std(self):
        return self.dataset.all_points_std

    @property
    def shuffle_idx(self):
        return self.dataset.shuffle_idx

    @property
    def all_points(self):
        return self.dataset.all_points

    @property
    def dataset_original_length(self):
        return len(self.dataset)


if __name__ == "__main__":
    shape_ds = ShapeNet15kPointClouds(root_dir='/pio/gluster/data/shapenet/ShapeNetCore.v2.PC15k', categories=['airplane'], split='val')
    print(next(iter(shape_ds)))