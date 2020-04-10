import random
import numpy as np
from torch import nn
import torch
import torch.sparse
from torch.distributions import Categorical
from state import State

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

def select_node (candidates, state, policy, partition_size, partition_edges):
    candidates_list = list(candidates)
    probabilities = policy.forward(candidates_list, state, partition_size, partition_edges)
    m = Categorical(probabilities)
    action = m.sample()
    return candidates_list[action.item()], m.log_prob(action)

def select_node_dummy (candidates, features, policy, partition_size, partition_edges):
    return select_random(candidates), torch.tensor([1.0])

def select_random (nodes):
    return random.choice(list(nodes))

def add_to_S(state, S, n):
    S.add(n)
    state.add_S(n)

def clear_S(state, S):
    state.clear_S()
    S.clear()

def mark_assigned(state, unassigned, n):
    unassigned.remove(n)
    state.add_assigned(n)


def get_features_embedding(graph, unassigned, S, candidates):
    num_nodes = len(graph.nodes)
    # neighborhood_nodes = set()
    # for u in candidates:
    #     neighborhood_nodes.add(u)
    #     for v in graph.adj_list[u]:
    #         neighborhood_nodes.add(v)
    #
    # first_hop = neighborhood_nodes
    # neighborhood_nodes = set()
    # for u in first_hop:
    #     neighborhood_nodes.add(u)
    #     for v in graph.adj_list[u]:
    #         neighborhood_nodes.add(v)
    #
    # index = []
    # vals = []
    # for u in neighborhood_nodes:
    #     vals.append(len(graph.adj_list[u]))
    #     index.append([u, 0])
    #     if u not in unassigned:
    #         vals.append(1)
    #         index.append([u, 1])
    #     if u in S :
    #         vals.append(1)
    #         index.append([u, 2])
    #
    # indices_tensor = torch.tensor(index)
    # weight = torch.sparse_coo_tensor(indices_tensor.t(), vals, [num_nodes, 3], dtype=torch.float)


    feat_data = np.zeros((num_nodes, 4))
    for n in range(num_nodes):
        feat_data[n][0] = len(graph.adj_list[n])
        if n not in unassigned:
            feat_data[n][1] = 1
        else:
            for v in graph.adj_list[n]:
                if v in unassigned and v not in S:
                    feat_data[n][3] += 1
        if n in S :
            feat_data[n][2] = 1


    features = nn.Embedding(num_nodes, 4)

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
    print("avg_rf {}\n".format(avg_rf))
    #assert avg_rf == -total_rewards

def run_episode(graph, policy, num_partitions):
    action_probs = []
    rewards = []

    num_edges = (len(graph.edges) / 2)
    partition_size = int(num_edges / num_partitions)
    if partition_size * num_partitions < num_edges:
        partition_size += 1  # achieve as much balancing as possible.

    partitions = {}
    state = State(graph)
    p = 0
    partitions[0] = []
    while p < num_partitions:
        last_assigned = None
        left_over_edges = []
        while len(partitions[p]) < partition_size:
            candidates = state.S.difference(state.C)
            if len(candidates) == 0:
                if len(state.unassigned) == 0:
                    break
                candidates.add(select_random(state.unassigned))

            #features = get_features_embedding(graph, state.unassigned, state.S, candidates)
            selected_node, action_prob = select_node(candidates, state, policy, partition_size, partitions[p])
            action_probs.append(action_prob)

            was_already_in_partition = True
            if selected_node not in state.S:
                was_already_in_partition = False  # need to know if RF of selected_node increased in this move.

            state.assign(selected_node)
            state.add_S(selected_node)
            state.add_C(selected_node)

            # collect the neighbors of the newly selected node u
            # make sure to put nodes already in S ahead of the nodes that are not.
            # This is because edges incident to nodes in S can be added without any additional cost, so in the case
            # where the partition doesn't have enough room for all u's edges, it is beneficial to add as many of the
            # edges incident to nodes in S as possible before starting a new partition and increasing the RF.
            neighbors = []
            for v in graph.adj_list[selected_node]:
                if v in state.unassigned and v in state.S:
                    neighbors.append(v)

            for v in graph.adj_list[selected_node]:
                if v in state.unassigned and (v not in state.S):
                    neighbors.append(v)

            # now calculate the incremental increase in RF as a result of this choice.
            reward = 0
            for v in neighbors:
                if len(partitions[p]) < partition_size:
                    if v not in state.S:
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
                state.add_S(v)

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
        state.clear_C()
        state.clear_S()

        # If a partition is not fully filled with left over edges, set up its state
        if len(left_over_edges) > 0:
            rewards[-1] -= 1  # account for the increase in RF of the core node
            state.add_C(last_assigned)
            state.add_S(last_assigned)

        for e in left_over_edges:
            state.add_S(e[1])

    rewards = [r / len(graph.nodes) for r in rewards]
    validate_reward(partitions, rewards, len(graph.nodes))
    return action_probs, rewards

