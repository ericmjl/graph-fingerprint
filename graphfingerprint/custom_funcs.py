import networkx as nx
import math
from random import sample


def make_random_graph(nodes, n_edges, features_dict):
    """
    Makes a randomly connected graph.
    """

    G = nx.Graph()
    for n in nodes:
        G.add_node(n, features=features_dict[n])

    for i in range(n_edges):
        u, v = sample(G.nodes(), 2)
        G.add_edge(u, v)

    return G


def score(g):
    """
    This score is the number of nodes in the graph.
    """
    return len(g.nodes())


def score_regressable(G):
    """
    The regressable score for each graph will be the sum of the
    (square root of each node + the sum of its neighbors.)
    """
    sum_score = 0
    for n, d in G.nodes(data=True):
        sum_score += math.sqrt(n)

        for nbr in G.neighbors(n):
            sum_score += nbr
    return sum_score


def get_graph_idxs(graphs):
    """
    Given a list of graphs, returns a list of row indices stored on each graph.
    """
    idxs = []
    for g in graphs:
        for n, d in g.nodes(data=True):
            idxs.extend(d['idx'])
    return idxs
