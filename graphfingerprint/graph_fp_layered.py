import networkx as nx
import math
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from random import sample, choice
from convnet import GraphInputLayer, GraphConvLayer, FingerprintLayer,\
    LinearRegressionLayer
from wb2 import WeightsAndBiases
from flatten import flatten
from time import time
from autograd import grad


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

# features_dict will look like this:
# {0: array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#  1: array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
#  2: array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
#  3: array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
#  4: array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
#  5: array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
#  6: array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
#  7: array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
#  8: array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
#  9: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}

all_nodes = [i for i in range(10)]
lb = LabelBinarizer()
features_dict = {i: lb.fit_transform(all_nodes)[i] for i in all_nodes}

G = make_random_graph(sample(all_nodes, 6), 5, features_dict)
G.edges(data=True)


# def score(G):
#     """
#     The regressable score for each graph will be the sum of the
#     (square root of each node + the sum of its neighbors.)
#     """
#     sum_score = 0
#     for n, d in G.nodes(data=True):
#         sum_score += math.sqrt(n)

#         for nbr in G.neighbors(n):
#             sum_score += nbr
#     return sum_score


def score(g):
    """
    This score is the number of nodes in the graph.
    """
    return len(g.nodes())

print('Score of the graph:')
print(score(G))

n_nodes = [i for i in range(2, len(all_nodes))]
n_graphs = 10

# Make all synthetic graphs
graphs = [make_random_graph(sample(all_nodes, choice(n_nodes)), 
                            choice(n_nodes), 
                            features_dict) for i in range(n_graphs)]

input_shape = (1, 10)
inputs = GraphInputLayer(input_shape).forward_pass(graphs)
node_indices, nbr_indices, graph_indices = \
    GraphInputLayer(input_shape).graph_indices(graphs)
# print('Node indices:')
# print(node_indices)
# print('Neighbor indices:')
# print(nbr_indices)
# print('Graph indices:')
# print(graph_indices)

layers = [GraphConvLayer(kernel_shape=(10, 20)),
          GraphConvLayer(kernel_shape=(20, 20)),
          GraphConvLayer(kernel_shape=(20, 10)),
          # GraphConvLayer(kernel_shape=(10, 10)),
          FingerprintLayer(10),
          LinearRegressionLayer(shape=(10, 1))]
print(layers)

def initialize_network(input_shape, layers, graphs):
    """
    Initializes all weights, biases and other parameters to random floats
    between 0 and 1.

    Returns a WeightsAndBiases class that stores all of the parameters
    as well.
    """
    wb_all = WeightsAndBiases()
    flatteners = dict()
    curr_shape = input_shape
    for i, layer in enumerate(layers):
        # print(i, layer, curr_shape)
        curr_shape, wb = layer.build_weights(curr_shape)
        # print(curr_shape)
        wb_all['layer{0}_{1}'.format(i, layer)] = wb

    return wb_all

wb_all = initialize_network(input_shape, layers, graphs)


def predict(wb_vect, wb_unflattener, inputs, layers, node_indices,
            nbr_indices, graph_indices):
    curr_inputs = inputs
    start = time()
    wb_all = wb_unflattener(wb_vect)
    end = time()
    # print('Time to unflatten: {0}'.format(end - start))
    for i, layer in enumerate(layers):
        wb = wb_all['layer{0}_{1}'.format(i, layer)]
        curr_inputs = layer.forward_pass(wb, curr_inputs, node_indices,
                                         nbr_indices, graph_indices)
    return curr_inputs

predict(*wb_all.flattened(), inputs, layers, node_indices, nbr_indices,
        graph_indices)

graph_scores = np.array([float(score(g)) for g in graphs])
print(type(graph_scores))

def train_loss(wb_vect, wb_unflattener, inputs, layers, node_indices,
               nbr_indices, graph_indices, graphs, graph_scores):
    """
    Training loss is MSE.
    """
    preds = predict(wb_vect, wb_unflattener, inputs, layers, node_indices,
                    nbr_indices, graph_indices)
    # actual = np.array([score(g) for g in graphs])

    mse = np.sum(np.power(preds - graph_scores, 2)) / len(graphs)
    # mse = np.sum(np.abs(preds - graph_scores)) / len(graphs)
    return mse

# Test the train_loss function
# train_loss(*wb_all.flattened(), inputs, layers, node_indices, nbr_indices,
#            graph_indices, graphs, graph_scores)

gradfunc = grad(train_loss)

# Test the gradfunc function.
# gradfunc(*wb_all.flattened(), inputs, layers, node_indices, nbr_indices,
#          graph_indices, graphs, graph_scores)

wb_vect, wb_unflattener = wb_all.flattened()


def sgd(gradfunc, wb_vect, wb_unflattener, inputs, layers, node_indices,
        nbr_indices, graph_indices, graphs, graph_scores, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """
    Stochastic gradient descent with momentum.
    """
    # wb_vect, wb_unflattener = wb.flattened()
    velocity = np.zeros(len(wb_vect))
    for i in range(num_iters):
        start = time()
        print('Epoch: {0}'.format(i))
        print('Computing gradient w.r.t. weights...')
        g = gradfunc(wb_vect, wb_unflattener, inputs, layers, node_indices,
                     nbr_indices, graph_indices, graphs, graph_scores)
        velocity = mass * velocity - (1.0 - mass) * g
        wb_vect += step_size * velocity
        print('Training Loss: ')
        print(train_loss(wb_vect, wb_unflattener, inputs, layers,
                         node_indices, nbr_indices, graph_indices, graphs,
                         graph_scores))
        end = time()
        print('Time: {0}'.format(end - start))
        print('')
    # return wb_vect, wb_unflattener

sgd(gradfunc, wb_vect, wb_unflattener, inputs, layers, node_indices,
    nbr_indices, graph_indices, graphs, graph_scores, num_iters=1000,
    step_size=0.0001)
