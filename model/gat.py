import torch
from torch import nn
from torch.nn import Dropout, LeakyReLU, Parameter, ELU, ModuleList
from torch.nn.init import xavier_uniform_


class GraphAttentionNetwork(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT."""
        super(GraphAttentionNetwork, self).__init__()

        self.attentions = ModuleList(
            [GraphAttentionLayer(n_feat,
                                 n_hid,
                                 dropout=dropout,
                                 alpha=alpha) for _ in range(n_heads)])

        self.out_att = GraphAttentionLayer(n_hid * n_heads,
                                           n_class,
                                           dropout=dropout,
                                           alpha=alpha)

        self.dropout = Dropout(dropout)
        self.elu = ELU()

    def forward(self, inputs):
        nodes, adj, idx = inputs
        x = self.dropout(nodes)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.out_att(x, adj)
        return x[idx]


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = Parameter(torch.zeros(size=(in_features, out_features)))
        xavier_uniform_(self.W.data, gain=1.414)
        self.a = Parameter(torch.zeros(size=(2 * out_features, 1)))
        xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = LeakyReLU(self.alpha)

    def forward(self, inputs, adj):
        h = torch.mm(inputs, self.W)
        n = h.size()[0]

        a_input = torch.cat([h.repeat(1, n).view(n * n, -1), h.repeat(n, 1)], dim=1).view(n, -1, 2 * self.out_features)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)

        return h_prime
