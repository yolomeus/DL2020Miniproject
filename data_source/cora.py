import gzip

import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset

from preprocessing.load_cora import load_data


class Cora(Dataset):
    """Dataset for loading cora.
    """

    def __init__(self, data_dir, mode='train'):
        data_dir = to_absolute_path(data_dir)
        adj, features, labels, idx_train, idx_val, idx_test = load_data(data_dir)

        self.adj = adj
        self.features = features
        self.labels = labels

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            idx = self.idx_train
        elif self.mode == 'val':
            idx = self.idx_val
        else:
            idx = self.idx_test

        return (self.features, self.adj, idx), self.labels[idx]

    def __len__(self):
        return 1

    @staticmethod
    def collate_fn(batch):
        return batch[0]
