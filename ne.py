import random

def select_random (nodes):
    return random.choice(list(nodes))

def select_node (candidates, S, graph, removed_edges):
    min_c = -1
    min_score = len(graph.nodes) + 1
    for c in candidates:
        score = 0
        for v in graph.adj_list[c]:
            if (c, v) not in removed_edges and v not in S:
                score += 1
        if score < min_score:
            min_c = c
            min_score = score

    return min_c

def calculate_rf(partitions, num_nodes):
    node_partitions = {}
    for p in partitions.keys():
        for (u, v) in partitions[p]:
            if u not in node_partitions:
                node_partitions[u] = set()
            if v not in node_partitions:
                node_partitions[v] = set()

            node_partitions[u].add(p)
            node_partitions[v].add(p)

    total_rf = 0
    for n in node_partitions.keys():
        total_rf += len(node_partitions[n])
    avg_rf = total_rf / num_nodes

def partition(graph, num_partitions):
    num_edges = (len(graph.edges) / 2)
    partition_size = int(num_edges / num_partitions)
    if partition_size * num_partitions < num_edges:
        partition_size += 1  # achieve as much balancing as possible.

    unassigned = set(graph.nodes)
    node_degrees = {}
    for n in graph.nodes:
        node_degrees[n] = len(graph.adj_list[n])
    partitions = {}
    removed_edges = set()

    for p in range(num_partitions):
        partitions[p] = []
        C = set()  # partition's core set
        S = set()  # partition's cover set

        while len(partitions[p]) < partition_size:
            candidates = S.difference(C)
            if len(candidates) == 0:
                if len(unassigned) == 0:
                    break
                candidates.add(select_random(unassigned))

            selected_node = select_node(candidates, S, graph, removed_edges)

            C.add(selected_node)
            S.add(selected_node)

            neighbors = [selected_node]
            for v in graph.adj_list[selected_node]:
                if v in unassigned and (v not in S):
                    neighbors.append(v)
                    S.add(v)

            for n in neighbors:
                for v in graph.adj_list[n]:
                    if len(partitions[p]) >= partition_size:
                        break
                    if v in S and (n, v) not in removed_edges:
                        partitions[p].append((n, v))
                        removed_edges.add((n, v))
                        removed_edges.add((v, n))
                        node_degrees[n] -= 1
                        if node_degrees[n] == 0:
                            unassigned.remove(n)
                        node_degrees[v] -= 1
                        if node_degrees[v] == 0:
                            unassigned.remove(v)

    calculate_rf(partitions, len(graph.nodes))








