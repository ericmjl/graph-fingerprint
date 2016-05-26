from autograd import numpy as np
from autograd.core import getval
import autograd.numpy.random as npr
from .wb import WeightsAndBiases
from collections import defaultdict
from .nonlinearity import relu


class GraphInputLayer(object):
    """
    Note: at each layer, we are only specifying the input shapes and output
    shapes.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "InputLayer"

    def forward_pass(self, graphs):
        """
        Returns the nodes' features stacked together, along with a dictionary
        of nodes and their neighbors.

        An example structure is:
        - {1: [1, 2 , 4],
           2: [2, 1],
           ...
           }
        """
        # First off, we label each node with the index of each node's data.
        features = []
        i = 0
        for g in graphs:
            for n, d in g.nodes(data=True):
                features.append(d['features'])
                g.node[n]['idx'] = i
                i += 1

        # We then do a second pass over the graphs, and record each node and
        # their neighbors' indices in the stacked features array.
        #
        # We also record the indices corresponding to each graph.
        nodes_nbrs = defaultdict(list)
        graph_idxs = defaultdict(list)
        for idx, g in enumerate(graphs):
            g.graph['idx'] = idx  # set the graph's index attribute.
            for n, d in g.nodes(data=True):
                nodes_nbrs[d['idx']].append(d['idx'])
                graph_idxs[idx].append(d['idx'])  # append node index to list
                                                  # of graph's nodes indices.
                for nbr in g.neighbors(n):
                    nodes_nbrs[d['idx']].append(g.node[nbr]['idx'])

        return np.vstack(features), nodes_nbrs, graph_idxs


class GraphConvLayer(object):
    """
    A graph convolution layer. Convolution operation is:

        [self + nbrs] (shape=(1row x n_feats)) @ weights + bias
    """
    def __init__(self, weights_shape, biases_shape):
        """
        Parameters:
        ===========
        """
        self.weights_shape = weights_shape
        self.biases_shape = biases_shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "GraphConvLayer"

    def forward_pass(self, wb, inputs, nodes_nbrs, graph_idxs):
        """
        Parameters:
        ===========
        - inputs: (np.array) the output from the previous layer.
        - graphs: (list) of nx.Graph objects.  TODO: Change to a dictionary of
                  {node: [self and neighbors]}
        """

        weights = wb['weights']
        biases = wb['biases']

        ####---------------------#####
        # # Old Version
        # npsum = np.sum
        # activations = np.zeros(shape=inputs.shape)
        # for n, nbrs in sorted(nodes_nbrs.items()):
        #     activations[n] = npsum(getval(inputs[nbrs]), axis=0)
        ####---------------------#####

        ####---------------------#####
        # New Version from @jakevdp
        new_nbrs = np.full((inputs.shape[0],
                            max(map(len, nodes_nbrs.values()))), -1, dtype=int)
        for i, v in nodes_nbrs.items():
            new_nbrs[i, :len(v)] = v
        # add a row of zeros to X
        new_inputs = np.vstack([inputs, 0 * inputs[0]])
        activations = new_inputs.take(new_nbrs, 0).sum(1)
        ####---------------------#####

        return relu(np.dot(activations, weights) + biases)

    def build_weights(self):
        """
        Parameters:
        ===========
        - input_shape: (2-tuple) of integers.

        Returns:
        ========
        - output_shape: (2-tuple) of integers specifying the output shape.
        """
        self.wb.add(name='weights', shape=self.weights_shape)
        # self.wb.add(name='nbr_weights', shape=self.kernel_shape)
        self.wb.add(name='biases', shape=self.biases_shape)

        return self.wb


class FingerprintLayer(object):
    def __init__(self, weights_shape, biases_shape):
        """
        Parameters:
        ===========
        - shape: int: the length of the fingerprint.
        """
        self.weights_shape = weights_shape
        self.biases_shape = biases_shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "FingerprintLayer"

    def forward_pass(self, wb, inputs, nodes_nbrs, graph_idxs):
        """
        Parameters:
        ===========
        - inputs: (np.array) the output from the previous layer, of shape
                  (n_all_nodes, n_features)
        - graphs: (list of nx.Graphs)
        """

        fingerprints = []
        for g, idxs in sorted(graph_idxs.items()):
            fp = np.sum(inputs[idxs], axis=0)
            fingerprints.append(fp)

        assert len(fingerprints) == len(graph_idxs)
        return np.vstack(fingerprints)

    def build_weights(self):
        """
        Builds the fingerprint. The shape of the fingerprint does not
        necessarily have to be the same as the initial feature vector.
        """
        self.wb.add('weights', shape=self.weights_shape)

        return self.wb


class FullyConnectedLayer(object):
    """
    A fully connected layer.
    """

    def __init__(self, weights_shape, biases_shape):
        self.weights_shape = weights_shape
        self.biases_shape = biases_shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "FullyConnectedLayer"

    def forward_pass(self, wb, inputs, nodes_nbrs, graph_idxs):
        return relu(np.dot(inputs, wb['weights']) + wb['bias'])

    def build_weights(self):
        self.wb.add('weights', shape=self.weights_shape)
        self.wb.add('bias', shape=self.biases_shape)
        return self.wb


class DropoutLayer(object):
    """
    A dropout layer randomly sets particular columns of the outputs to be
    zeros with a probability of p.
    """
    def __init__(self, p):
        self.p = p
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "DropoutLayer"

    def forward_pass(self, wb, inputs, nodes_nbrs, graph_idxs):
        return inputs * npr.binomial(1, self.p, size=(inputs.shape))

    def build_weights(self):
        self.wb.add('weights', shape=(1, 1))
        return self.wb


class LinearRegressionLayer(object):
    """
    Linear regression layer.
    """
    def __init__(self, weights_shape, biases_shape):
        self.weights_shape = weights_shape
        self.biases_shape = biases_shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "LinearRegressionLayer"

    def forward_pass(self, wb, inputs, nodes_nbrs, graph_idxs):
        return np.dot(inputs, wb['linweights']) + wb['bias']

    def build_weights(self):
        self.wb.add('linweights', shape=self.weights_shape)
        self.wb.add('bias', shape=self.biases_shape)

        return self.wb
