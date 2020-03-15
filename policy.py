import torch
import torch.nn as nn
from torch.nn import init
from graphsage.aggregators import MeanAggregator
from graphsage.encoders import Encoder

class Policy(nn.Module):
    def __init__(self, adj_lists, cuda=False):
        super(Policy, self).__init__()
        self.cuda = cuda

        agg1 = MeanAggregator(cuda=cuda)
        enc1 = Encoder(3, 128, adj_lists, agg1, gcn=True, cuda=cuda)
        agg2 = MeanAggregator(cuda=cuda)
        enc2 = Encoder(enc1.embed_dim, 128, adj_lists, agg2,
                       base_model=enc1, gcn=True, cuda=False)
        enc1.num_samples = 50
        enc2.num_samples = 10
        self.enc1 = enc1
        self.GCN = enc2

        # Fully connected network that outputs node probabilities.
        input_dim = 128 + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.smax = nn.Softmax(dim=0)

    def forward(self, nodes, features, partition_size, partition_edges):
        embeddings = self.GCN.forward(nodes, lambda nodes: self.enc1(nodes, features).t()).t()
        remaining_space = partition_size - len(partition_edges)
        #requires grad?
        partition_column = torch.Tensor(len(nodes), 1)
        partition_column.fill_(remaining_space)
        all_input = torch.cat((embeddings, partition_column), 1)
        scores = self.network.forward(all_input).squeeze(dim=1)
        return self.smax.forward(scores)




