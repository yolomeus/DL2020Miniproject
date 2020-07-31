import torch
from torch import nn
from torch.nn import Dropout, LeakyReLU, Parameter, ELU, ModuleList, Linear
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_


class GraphAttentionNeighbourNetwork(nn.Module):
    """
    GAT attending over higher order neighbours.
    """

    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads, n_orders, n_nodes, method, graph_convolve):
        super().__init__()

        assert n_heads >= n_orders
        self.n_orders = n_orders
        self.n_heads = n_heads
        self.method = method
        assert method in ['distributed', 'single']

        self.attentions = ModuleList(
            [GraphAttentionLayer(n_feat,
                                 n_hid,
                                 dropout=dropout,
                                 alpha=alpha,
                                 graph_convolve=graph_convolve) for _ in range(n_heads)])

        self.out_att = GraphAttentionLayer(n_hid * n_heads,
                                           n_class,
                                           dropout=dropout,
                                           alpha=alpha,
                                           graph_convolve=graph_convolve)

        self.dropout = Dropout(dropout)
        self.elu = ELU()

    def forward(self, inputs):
        nodes, adj, idx = inputs
        if self.training:
            nodes = nodes[idx]
            adj = adj[idx]
            adj = adj[:, idx]

        adj[adj != 0] = 1

        x = self.dropout(nodes)
        att_outs = []
        if self.method == 'distributed':
            # individual heads attend over different order neighbourhoods
            adj_matrices = [normalize(adj, p=1)]
            nth_adj = adj
            for _ in range(self.n_orders):
                nth_adj = nth_adj @ adj
                adj_matrices.append(normalize(nth_adj, p=1))

            for i in range(self.n_heads):
                nth_adj = adj_matrices[i % (self.n_orders + 1)]
                att_out = self.attentions[i](x, nth_adj)
                att_outs.append(att_out)

        elif self.method == 'single':
            # compute n-th order reachability matrix
            nth_adj = adj
            for _ in range(self.n_orders):
                nth_adj = nth_adj @ adj + nth_adj
            nth_adj = normalize(nth_adj, p=1)
            # every head attends over the same neighbourhood
            for head in self.attentions:
                att_outs.append(head(x, nth_adj))
        else:
            raise NotImplementedError()

        x = torch.cat(att_outs, dim=1)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.out_att(x, adj)

        # only return test instances when testing
        return x if self.training else x[idx]


class GraphAttentionNetwork(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT."""
        super(GraphAttentionNetwork, self).__init__()

        self.attentions = ModuleList(
            [GraphAttentionLayer(n_feat,
                                 n_hid,
                                 dropout=dropout,
                                 alpha=alpha,
                                 graph_convolve=False) for _ in range(n_heads)])

        self.out_att = GraphAttentionLayer(n_hid * n_heads,
                                           n_class,
                                           dropout=dropout,
                                           alpha=alpha,
                                           graph_convolve=False)

        self.dropout = Dropout(dropout)
        self.elu = ELU()

    def forward(self, inputs):
        nodes, adj, idx = inputs
        # when training, only use the training sub-graph (inductive)
        if self.training:
            nodes = nodes[idx]
            adj = adj[idx]
            adj = adj[:, idx]

        x = self.dropout(nodes)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.out_att(x, adj)

        # only return test instances when testing
        return x if self.training else x[idx]


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, graph_convolve):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.graph_convolve = graph_convolve

        if graph_convolve:
            self.W_gc = Parameter(torch.zeros(size=(in_features, out_features)))
            self.b_gc = Parameter(torch.zeros(out_features))
            xavier_uniform_(self.W_gc.data, gain=1.414)

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

        if self.graph_convolve:
            gc = inputs @ self.W_gc
            gc = adj @ gc + self.b_gc
            h_prime += gc

        return h_prime
