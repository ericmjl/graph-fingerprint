import networkx as nx
import math
import autograd.numpy as np
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


def score_sqrt(G):
    """
    The regressable score for each graph will be the sum of the
    (square root of each node + its neighbors.)
    """
    sum_score = 0
    for n, d in G.nodes(data=True):
        sum_score += math.sqrt(n)

        for nbr in G.neighbors(n):
            sum_score += math.sqrt(nbr)
    return sum_score


def score_nbr(G):
    """
    This regression score is the sum of itself + neighbors.
    """
    sum_score = 0
    for n, d in G.nodes(data=True):
        sum_score += n

        for nbr in G.neighbors(n):
            sum_score += nbr

    return sum_score


def score_sum(G):
    """
    This is the sum of nodes.
    """
    return sum([n for n in G.nodes()])


def score_random(G):
    """
    Returns a random number between 0 and 10 for the score. This is meant to
    be a null learning task, where the weights and biases should never
    converge.
    """
    return np.random.randint(0, 10)


def score_edges(G):
    """
    Pinpoints certain edges as being important.
    """
    important_edges = [(1, 2), (2, 5), (3, 4)]

    score = 0
    for edge in important_edges:
        if G.has_edge(*edge):
            score += sum(edge)
    return score


def score_sine(G):
    """
    Takes advantage of the non-linear "sine" function.
    Structure similar to score_nbr, but applies sine function at each addition.
    """
    sum_score = 0
    for n in G.nodes():
        sum_score += np.sin(n)
        for nbr in G.neighbors(n):
            sum_score += np.sin(nbr)

    return sum_score


def score_tan(G):
    """
    Takes advantage of tangent function.
    """

    sum_score = 0
    for n in G.nodes():
        sum_score += np.tan(n)
        for nbr in G.neighbors(n):
            sum_score += np.tan(nbr)

    return sum_score

def get_graph_idxs(graphs):
    """
    Given a list of graphs, returns a list of row indices stored on each graph.
    """
    idxs = []
    for g in graphs:
        idxs.exend(graph_indices(g))
    return idxs


def graph_indices(g):
    """
    Returns the row indices of each of the nodes in the graphs.
    """
    return [d['idx'] for _, d in g.nodes(data=True)]
