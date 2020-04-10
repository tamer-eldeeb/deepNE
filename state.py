import numpy as np
import torch

class State:
    def __init__(self, graph):
        self.graph = graph
        self.assigned = set()
        self.unassigned = set(graph.nodes)
        self.S = set()
        self.C = set()

    def assign(self, n):
        self.assigned.add(n)
        self.unassigned.remove(n)

    def add_C(self, n):
        self.C.add(n)

    def clear_C(self):
        self.C.clear()

    def add_S(self, n):
        self.S.add(n)

    def clear_S(self):
        self.S.clear()

    def get_features_tensor(self, nodes):
        num_nodes = len(nodes)
        feat_data = np.zeros((num_nodes, 4))
        for i in range(num_nodes):
            n = nodes[i]
            feat_data[i][0] = len(self.graph.adj_list[n])
            if n in self.assigned:
                feat_data[i][1] = 1
            else:
                for v in self.graph.adj_list[n]:
                    if v not in self.assigned and v not in self.S:
                        feat_data[i][3] += 1
            if n in self.S:
                feat_data[i][2] = 1

        return torch.FloatTensor(feat_data)

    def __call__(self, nodes):
        return self.get_features_tensor(nodes)
