import pickle

import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


class Cora(Dataset):
    """Dataset for loading cora.
    """

    def __init__(self, data_path, mode='train'):
        data_path = to_absolute_path(data_path)
        with open(data_path, 'rb') as fp:
            adj, features, labels, idx_train, idx_val, idx_test = map(torch.as_tensor, pickle.load(fp))

        self.adj = adj.to(torch.float32)
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
        elif self.mode == 'test':
            idx = self.idx_test
        else:
            raise NotImplementedError()

        return (self.features, self.adj, idx), self.labels[idx]

    def __len__(self):
        return 1

    @staticmethod
    def collate_fn(batch):
        return batch[0]
