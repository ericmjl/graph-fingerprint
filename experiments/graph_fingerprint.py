import math
from random import sample, choice

import autograd.numpy as np
import networkx as nx
from autograd import grad
from autograd.scipy.misc import logsumexp
from sklearn.preprocessing import LabelBinarizer

from wb import WeightsAndBiases

wb = WeightsAndBiases(n_layers=2, shapes=(10, 20, 10))

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
features_dict = {i:lb.fit_transform(all_nodes)[i] for i in all_nodes}

G = make_random_graph(sample(all_nodes, 6), 5, features_dict)
G.edges(data=True)


def score(G):
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

score(G)


G.nodes(data=True)[0][1]['features'].shape



def softmax(X, axis=0):
    """
    The softmax function normalizes everything to between 0 and 1.
    """
    return np.exp(X - logsumexp(X, axis=axis, keepdims=True))

# test softmax:
X = np.random.random((1,10))
softmax(X, axis=1)



def relu(X):
    """
    The ReLU - Rectified Linear Unit.
    """
    return X * (X > 0)


# # test relu:
# X = np.random.normal(0, 1, size=(5, 1))
# print(X)
# print('')
# print(relu(X))

# Make 1000 random graphs.
syngraphs = []
for i in range(1000):
    n_nodes = choice([i for i in range(2, 10)])
    n_edges = choice([i for i in range(1, n_nodes**2)])
    
    G = make_random_graph(sample(all_nodes, n_nodes), n_edges, features_dict)
    syngraphs.append(G)
    
len(syngraphs)


# Write a function that computes the feature matrix, and writes the
# indices to the nodes of each graph.
def stacked_node_activations(graphs):
    """
    Note: this function should only be called for computing the
    stacked node activations after initializing the graphs.
    
    Inputs:
    =======
    - graphs: (list) a list of graphs on which to stack their
              feature vectors.
    """
    features = []
    curr_idx = 0
    for g in graphs:
        for n, d in g.nodes(data=True):
            features.append(d['features'])
            g.node[n]['idx'] = curr_idx
            curr_idx += 1
    return np.stack(features)

# # test stacked_node_activations
layers = dict()
layers[0] = stacked_node_activations(syngraphs)
print('testing stacked_node_activations')
print(layers[0])

# Write a function that gets the indices of each node's neighbors.
def neighbor_indices(G, n):
    """
    Inputs:
    =======
    - G: the graph to which the node belongs to.
    - n: the node inside the graph G.
    
    Returns:
    ========
    - indices: (list) a list of indices, which should (but is not
               guaranteed to) correspond to a row in a large 
               stacked matrix of features.
    """
    indices = []
    for n in G.neighbors(n):
        indices.append(G.node[n]['idx'])
    return indices


# test neighbor_indices
# nbr_idxs = neighbor_indices(syngraphs[0], syngraphs[0].nodes()[0])
# print('testing neighbor_indices')
# print(nbr_idxs)


# Write a function that sums each of the neighbors' activations for a
# given node in a given graph.
def neighbor_activations(G, n, activations_dict, layer):
    """
    Inputs:
    =======
    - G: the graph to which the node belongs to.
    - n: the node inside the graph G
    - activations_dict: a dictionary that stores the node activations 
                        at each layer.
    - layer: the layer at which to compute neighbor activations.
    """
    nbr_indices = neighbor_indices(G, n)
    return np.sum(activations_dict[layer][nbr_indices], axis=0)

# print('testing neighbor_activations')
# print(neighbor_activations(syngraphs[0], syngraphs[0].nodes()[0], layers, 0))


# Write a function that stacks each of the nodes' neighbors
# activations together into a large feature matrix.

def stacked_neighbor_activations(graphs, activations_dict, layer):
    """
    Inputs:
    =======
    - graphs: (list) a list of NetworkX graphs.
    - activations_dict: (dict) a dictionary where keys are the layer
                        number and values are the node activations.
    
    Returns:
    ========
    - a stacked numpy array of neighbor activations
    """
    nbr_activations = []
    for g in graphs:
        for n in g.nodes():
            nbr_acts = neighbor_activations(g, n, activations_dict, layer)
            nbr_activations.append(nbr_acts)
    return np.stack(nbr_activations)

# print('testing stacked_neighbor_activations')
# print(stacked_neighbor_activations(syngraphs, layers, 0))

# Write a function that computes the next layers' activations.

def activation(activations_dict, wb, layer, graphs):
    """
    Inputs:
    =======
    - activations_dict: (dict) a dictionary where keys are the layer
                        number and values are the node activations.
    - wb: (wb.WeightsAndBiases) the WB class storing the weights and
          biases.
    - layer: (int) the layer for which to compute the activations.    
    
    Returns:
    ========
    - a stacked numpy array of activations, which can be assigned to
      the activations_dict's next layer if desired (actually it
      should be).
    """
    
    self_acts = activations_dict[layer]
    self_acts = np.dot(self_acts, wb[layer]['self_weights'])

    nbr_acts = stacked_neighbor_activations(graphs, activations_dict, layer)
    nbr_acts = np.dot(nbr_acts, wb[layer]['nbr_weights'])
    # nbr_acts = nbr_acts.astype('float64')
    print(nbr_acts.dtype)
    
    biases = wb[layer]['biases']
    return relu(self_acts + nbr_acts + biases)

# test activation function
acts = activation(layers, wb, 0, syngraphs)

# Write a function that gets the indices of all of the nodes in the
# graph.
def graph_indices(g):
    """
    Returns the row indices of each of the nodes in the graphs.
    """
    return [d['idx'] for _, d in g.nodes(data=True)]

# Write a function that makes the fingerprint used for predictions.
def fingerprint(activations_dict, graphs):
    """
    Computes the final layer fingerprint for each graph.
    
    Inputs:
    =======
    - activations_dict: (dict) a dictionary where keys are the layer
                        number and values are the node activations.
    - graphs: a list of graphs for which to compute the fingerprints.
    
    Returns:
    ========
    - a stacked numpy array of fingerprints, of length len(graphs).
    """
    top_layer = max(activations_dict.keys())
    fingerprints = []
    for g in graphs:
        idxs = graph_indices(g)
        fp = np.sum(activations_dict[top_layer][idxs], axis=0)
        fingerprints.append(softmax(fp))

    return np.stack(fingerprints)

# test fingerprint function
fingerprint(layers, syngraphs)


# Write a function that makes the forward pass predictions.
def predict(wb_vect, wb_unflattener, activations_dict, graphs):
    """
    Makes predictions.
    
    Change this function for each new learning problem.
    
    Inputs:
    =======
    - wb_vect: (WeightsAndBiases.vect)
    - wb_unflattener (WeightsAndBiases.unflattener)
    - activations_dict: (dict) a dictionary where keys are the layer
                        number and values are the node activations.
    - graphs: a list of graphs for which to compute the fingerprints.
    
    Returns:
    ========
    - a numpy array of predictions, of length len(graphs).
    """
    
    wb = wb_unflattener(wb_vect)
    for k in sorted(wb.keys()):
        activations_dict[k + 1] = activation(activations_dict, wb, k, graphs)
        
    
    top_layer = max(wb.keys())
    
    fps = fingerprint(layers, graphs)
    
    return np.dot(fps, wb[top_layer]['linweights'])

print(predict(wb.vect, wb.unflattener, layers, syngraphs))


# Write a function that computes the training loss.
def train_loss(wb_vect, wb_unflattener, activations_dict, graphs):
    """
    Computes the training loss as mean squared error.
    
    Inputs:
    =======
    - wb_vect: (WeightsAndBiases.vect)
    - wb_unfalttener (WeightsAndBiases.unflattener)
    - activations_dict: (dict) a dictionary where keys are the layer
                        number and values are the node activations.
    - graphs: a list of graphs for which to compute the fingerprints.

    Returns:
    ========
    - mean squared error.
    """
    
    scores = np.array([score(g) for g in graphs]).reshape((len(graphs), 1))
    
    preds = predict(wb_vect, wb_unflattener, activations_dict, graphs)
    
    return np.sum(np.power(preds - scores, 2)) / len(scores)

print('testing train_loss')
tl = train_loss(wb.vect, wb.unflattener, layers, syngraphs)
print(tl)


gradfunc = grad(train_loss, argnum=0)
print('testing gradient function')
gradfunc(wb.vect, wb.unflattener, layers, syngraphs)