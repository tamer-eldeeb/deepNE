import episode_runner
from episode_runner import Graph
import numpy as np
import random
from policy import Policy
import torch
from torch import optim
import ne

def load_graph_edges(edge_file):
    edges = set()
    f = open(edge_file, "r")
    for l in f:
        nodes = l.split(" ")
        u = int(nodes[0])
        v = int(nodes[1].strip('\n'))
        edges.add((u, v))
        edges.add((v, u))
    return list(edges)

def test():
    #np.random.seed(1)
    #random.seed(1)
    edges = load_graph_edges("email-Eu-core.txt")
    graph = Graph(edges)
    reinforce(graph, 10)
    #ne.partition(graph, 10)


def reinforce(graph, num_partitions):
    policy = Policy(graph.adj_list, cuda=False)

    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    num_epochs = 50
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        num_episodes = 10
        ep_probabilities = []
        ep_rewards = []
        for episode in range(num_episodes):
            log_probs, rewards = episode_runner.run_episode(graph, policy, num_partitions)
            ep_probabilities.append(log_probs)
            cum_rewards = []
            cum = 0
            for r in rewards:
                cum += r
                cum_rewards.append(cum)
            cum_rewards.reverse()
            ep_rewards.append(cum_rewards)

        # calculate reward baselines
        b = []
        T = len(graph.nodes)
        for t in range(T):
            running_sum = 0
            for episode in range(num_episodes):
                if t < len(ep_rewards[episode]):
                    running_sum += ep_rewards[episode][t]
            b.append(running_sum / num_episodes)
        print("baseline after epoch {}: {}", epoch, b[0])

        objective = torch.tensor([0.0], requires_grad=True)
        for episode in range(num_episodes):
            for t in range(len(ep_probabilities[episode])):
                objective = torch.add(objective, ep_probabilities[episode][t] * (ep_rewards[episode][t] - b[t]))
        objective = torch.mul(objective, -1)  # need to maximize the reward, which is equivalent to minimizing its -ve.
        print("objective {}", objective.item())
        objective.backward()
        optimizer.step()


if __name__ == "__main__":
    test()



