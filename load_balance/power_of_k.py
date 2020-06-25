import random
import numpy as np

def run(graph, num_partitions, seed):
    random.seed(seed)
    partition = {}
    for i in range(0, num_partitions):
        partition[i] = []
    partition_loads = np.zeros((num_partitions)) 
    print(len(graph))
    while len(graph):
        pos = random.randint(0, len(graph) - 1)
        task_id, load = graph[pos]
        partition_id = np.argmin(partition_loads)
        partition[partition_id].append(task_id)
        partition_loads[partition_id] += load
        del graph[pos]
    
    print(partition_loads)

