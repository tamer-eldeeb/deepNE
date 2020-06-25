import argparse
import power_of_k

parser = argparse.ArgumentParser(description='Load balance library')
parser.add_argument('--method', type=str, default='power_of_k',
                    help='the testing method (default: power_of_k)')

parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')

parser.add_argument('--num_partitions', type=int, default=10, metavar='N',
                    help='patitions number (default: 10)')

parser.add_argument('--graph_source', type=str, default='../email-Eu-core.txt',
                    help='the testing graph file')
args = parser.parse_args()

def create_workload(edges):
    dic = {}
    for edge in edges:
        (u, v) = edge
        if u not in dic:
            dic[u] = 1
        else:
            dic[u] += 1
    res = []
    for key in dic:
        res.append((key, dic[key]))
    return res




if __name__ == '__main__':
    loop = 0
    edges = set()
    with open(args.graph_source, 'r') as g:
        for line in g:
            nodes = line.split(" ")
            u = int(nodes[0])
            v = int(nodes[1].strip('\n'))
            if u == v:
                loop += 1
                continue
            edges.add((u, v))
            edges.add((v, u))
    if loop:
        print("Please notice: there exists ", loop, "loop edges in the given graph.")
    edges = list(edges)
    graph = create_workload(edges)

    power_of_k.run(graph, args.num_partitions, args.seed)



