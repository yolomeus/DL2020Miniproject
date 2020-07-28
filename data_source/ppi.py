import pickle

import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


class PPI(Dataset):
    """Dataset for loading PPI - Protein Protein interaction.
    """

    def __init__(self, data_path):
        data_path = to_absolute_path(data_path)
        with open(data_path, 'rb') as fp:
            x = pickle.load(fp)
            idx_mask = list(map(torch.as_tensor, x[-1]))
            adj, features, labels, = map(torch.as_tensor, x[:-1])

        self.adj = adj.to(torch.float32)
        self.features = features.to(torch.float32)
        self.labels = labels

        self.idx_masks = idx_mask

    def __getitem__(self, index):
        adj = self.adj[index]
        features = self.features[index]
        idx_mask = self.idx_masks[index]

        return (features, adj, idx_mask), self.labels[index]

    def __len__(self):
        return len(self.adj)

    # @staticmethod
    # def collate_fn(batch):
    #     graphs, labels = zip(*batch)
    #     return graphs, labels
