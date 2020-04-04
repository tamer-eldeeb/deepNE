import random
import numpy as np
from torch import nn
import torch
from torch.distributions import Categorical

class Graph:
    def __init__(self, edges):
        self.nodes = set()
        self.adj_list = {}
        self.edges = edges

        for (u, v) in edges:
            self.nodes.add(u)
            self.nodes.add(v)

            if u not in self.adj_list.keys():
                self.adj_list[u] = set()

            self.adj_list[u].add(v)

def select_node (candidates, features, policy, partition_size, partition_edges):
    candidates_list = list(candidates)
    probabilities = policy.forward(candidates_list, features, partition_size, partition_edges)
    m = Categorical(probabilities)
    action = m.sample()
    return candidates_list[action.item()], m.log_prob(action)

def select_node_dummy (candidates, features, policy, partition_size, partition_edges):
    return select_random(candidates), torch.tensor([1.0])

def select_random (nodes):
    return random.choice(list(nodes))


def get_features_embedding(graph, unassigned, S):
    num_nodes = len(graph.nodes)
    feat_data = np.zeros((num_nodes, 3))
    for n in range(num_nodes):
        feat_data[n][0] = len(graph.adj_list[n])
        if n not in unassigned:
            feat_data[n][1] = 1
        if n in S :
            feat_data[n][2] = 1

    features = nn.Embedding(num_nodes, 3)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    return features


def validate_reward(partitions, rewards, num_nodes):
    # Make sure the total rewards match the RF
    node_partitions = {}
    for p in partitions.keys():
        for (u, v) in partitions[p]:
            if u not in node_partitions:
                node_partitions[u] = set()
            if v not in node_partitions:
                node_partitions[v] = set()

            node_partitions[u].add(p)
            node_partitions[v].add(p)

    total_rewards = 0
    for r in rewards:
        total_rewards += r

    total_rf = 0
    for n in node_partitions.keys():
        total_rf += len(node_partitions[n])
    avg_rf = total_rf / num_nodes

    #assert total_rf == -total_rewards

def run_episode(graph, policy, num_partitions):
    action_probs = []
    rewards = []

    num_edges = (len(graph.edges) / 2)
    partition_size = int(num_edges / num_partitions)
    if partition_size * num_partitions < num_edges:
        partition_size += 1  # achieve as much balancing as possible.

    unassigned = set(graph.nodes)
    partitions = {}

    p = 0
    partitions[0] = []
    C = set()  # partition's core set
    S = set()  # partition's cover set
    while p < num_partitions:
        last_assigned = None
        left_over_edges = []
        while len(partitions[p]) < partition_size:
            candidates = S.difference(C)
            if len(candidates) == 0:
                if len(unassigned) == 0:
                    break
                candidates.add(select_random(unassigned))

            features = get_features_embedding(graph, unassigned, S)
            selected_node, action_prob = select_node(candidates, features, policy, partition_size, partitions[p])
            action_probs.append(action_prob)

            was_already_in_partition = True
            if selected_node not in S:
                was_already_in_partition = False  # need to know if RF of selected_node increased in this move.

            unassigned.remove(selected_node)
            C.add(selected_node)
            S.add(selected_node)

            # collect the neighbors of the newly selected node u
            # make sure to put nodes already in S ahead of the nodes that are not.
            # This is because edges incident to nodes in S can be added without any additional cost, so in the case
            # where the partition doesn't have enough room for all u's edges, it is beneficial to add as many of the
            # edges incident to nodes in S as possible before starting a new partition and increasing the RF.
            neighbors = []
            for v in graph.adj_list[selected_node]:
                if v in unassigned and v in S:
                    neighbors.append(v)

            for v in graph.adj_list[selected_node]:
                if v in unassigned and (v not in S):
                    neighbors.append(v)

            # now calculate the incremental increase in RF as a result of this choice.
            reward = 0
            for v in neighbors:
                if len(partitions[p]) < partition_size:
                    if v not in S:
                        reward -= 1
                    partitions[p].append((selected_node, v))
                else:
                    reward -= 1
                    last_assigned = selected_node
                    left_over_edges.append((selected_node, v))

            if len(neighbors) > 0 and not was_already_in_partition:
                reward -= 1  # the RF of selected_node has also increased.

            rewards.append(reward)

            for v in neighbors:
                S.add(v)

        # The algorithm relies on the assumption that once a node becomes "core" all its edges are assigned. This way
        # we don't need to keep track of individual edge state.
        # Handle the case where the partition fills up before adding all incident edges of a core set. Simply keep
        # filling as many partitions as necessary with its unassigned edges.
        while len(left_over_edges) > partition_size:
            rewards[-1] -= 1  # account for the increase in RF of the core node
            p = p + 1
            partitions[p] = list(left_over_edges[:partition_size-1])
            left_over_edges = left_over_edges[partition_size:]

        p = p + 1
        partitions[p] = left_over_edges
        C = set()
        S = set()

        # If a partition is not fully filled with left over edges, set up its state
        if len(left_over_edges) > 0:
            rewards[-1] -= 1  # account for the increase in RF of the core node
            C.add(last_assigned)
            S.add(last_assigned)

        for e in left_over_edges:
            S.add(e[1])

    rewards = [r / len(graph.nodes) for r in rewards]
    validate_reward(partitions, rewards, len(graph.nodes))
    return action_probs, rewards
