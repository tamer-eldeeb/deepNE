import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, state, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = state(unique_nodes_list).cuda()
        else:
            embed_matrix = state(unique_nodes_list)
        to_feats = mask.mm(embed_matrix)
        return to_feats


class LSTMAggregator(nn.Module):
    """
    Aggregates a node's embeddings using LSTM with a random permutation of the neighbors as input
    Some of this code is copied from: https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd
    with modification.
    """
    def __init__(self, input_dim, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(LSTMAggregator, self).__init__()

        self.cuda = cuda
        self.gcn = gcn
        self.hidden_size = input_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True)

    def forward(self, nodes, state, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        # will be used to pad all neighbor tensors to the same length
        max_neigh = 0
        for l in samp_neighs:
            max_neigh = max(max_neigh, len(l))

        num_nodes = len(nodes)

        node_neighbor_lengths = [len(samp_neigh) for _, samp_neigh in enumerate(samp_neighs)]
        node_neighbor_features = [state(list(samp_neigh), max_neigh) for _, samp_neigh in enumerate(samp_neighs)]
        lstm_input = torch.stack(node_neighbor_features)
        lstm_input = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, node_neighbor_lengths, batch_first=True, enforce_sorted=False)

        hidden = (torch.randn(1, num_nodes, self.hidden_size), torch.randn(1, num_nodes, self.hidden_size))
        out, _ = self.lstm(lstm_input, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out[:, -1, :]  # !! Taking final state, but could do something better (eg attention)
        return out


