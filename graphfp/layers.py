from autograd import numpy as np
import autograd.numpy.random as npr
from .wb2 import WeightsAndBiases
from collections import defaultdict
from .custom_funcs import graph_indices
from .nonlinearity import tanh, relu


class GraphInputLayer(object):
    """
    Note: at each layer, we are only specifying the input shapes and output
    shapes.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def __repr__(self):
        return "InputLayer"

    def graph_indices(self, graphs):
        """
        Returns a list of indices of each of the graph's nodes, and a
        list of indices of each of the graph's node's neighbors.

        There are two lists:
        1. A list of nodes' indices.
        2. A list of nodes' neighbors' indices.

        With the addition of this function, we can avoid doing iteration on
        graphs on each pass, thus speeding up each epoch.
        """
        node_idxs = []
        nbr_idxs = dict()
        graph_idxs = defaultdict(list)
        for i, g in enumerate(graphs):
            for n, d in g.nodes(data=True):
                node_idxs.append(g.node[n]['idx'])
                graph_idxs[i].append(g.node[n]['idx'])

                nbrs = []
                for nbr in g.neighbors(n):
                    nbrs.append(g.node[nbr]['idx'])
                nbr_idxs[g.node[n]['idx']] = nbrs
        return node_idxs, nbr_idxs, graph_idxs

    def forward_pass(self, graphs):
        """
        Returns the nodes' features stacked together.
        """
        features = []
        i = 0
        for g in graphs:
            for n, d in g.nodes(data=True):
                features.append(d['features'])
                g.node[n]['idx'] = i
                i += 1

        return np.vstack(features)

    def build_weights(self, input_shape):
        """
        Returns the output shape.
        """
        output_shape = self.forward_pass().shape

        return output_shape, self.wb


class GraphConvLayer(object):
    """
    A graph convolution layer. Convolution operation is:

      node_activations @ node_weights + nbr_activations @ nbr_weights + bias
    """
    def __init__(self, kernel_shape):
        """
        Parameters:
        ===========
        - kernel_shape: (2-tuple) of n_rows, n_cols. n_rows should correspond
                        to the number of columns for the matrix returned in
                        the previous layer.
        - nonlinearity: a nonlinearity function from the nonlinearity module.
        """
        self.kernel_shape = kernel_shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "GraphConvLayer"

    def forward_pass(self, wb, inputs, graphs):
        """
        Parameters:
        ===========
        - inputs: (np.array) the output from the previous layer.
        - graphs: (list) of nx.Graph objects.
        """
        def stacked_neighbor_activations(inputs, graphs):
            """
            Inputs:
            =======
            - graphs: (list) a list of NetworkX graphs.
            - inputs: (np.array) the inputs from the previous layer.

            Returns:
            ========
            - a stacked numpy array of neighbor activations
            """
            nbr_activations = []

            for g in graphs:
                for n in g.nodes():
                    nbr_acts = neighbor_activations(g, n, inputs)
                    nbr_activations.append(nbr_acts)

            return np.vstack(nbr_activations)

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

        def neighbor_activations(G, n, inputs):
            """
            Inputs:
            =======
            - G: the graph to which the node belongs to.
            - n: the node inside the graph G
            - inputs: the outputs from the previous layer.
            """
            nbr_indices = neighbor_indices(G, n)
            # print(nbr_indices)
            return np.sum(inputs[nbr_indices], axis=0)

        weights = wb['weights']
        biases = wb['biases']

        self_act = np.dot(inputs, weights)
        nbr_act = np.dot(stacked_neighbor_activations(inputs, graphs),
                         weights)
        # print('Computing activations...')
        return relu(self_act + nbr_act + biases)

    def build_weights(self, input_shape):
        """
        Parameters:
        ===========
        - input_shape: (2-tuple) of integers.

        Returns:
        ========
        - output_shape: (2-tuple) of integers specifying the output shape.
        """
        # assert input_shape[1] == self.kernel_shape[0],\
        #     'input_shape dim 1 must be same as kernel_shape dim 0'

        self.wb.add(name='weights', shape=self.kernel_shape)
        # self.wb.add(name='nbr_weights', shape=self.kernel_shape)
        self.wb.add(name='biases', shape=(1, self.kernel_shape[1]))

        output_shape = (input_shape[0], self.kernel_shape[1])

        return output_shape, self.wb


class FingerprintLayer(object):
    def __init__(self, shape):
        """
        Parameters:
        ===========
        - shape: int: the length of the fingerprint.
        """
        self.shape = shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "FingerprintLayer"

    def forward_pass(self, wb, inputs, graphs):
        """
        Parameters:
        ===========
        - inputs: (np.array) the output from the previous layer, of shape
                  (n_all_nodes, n_features)
        - graphs: (list of nx.Graphs)
        """

        fingerprints = []
        for g in graphs:
            idxs = graph_indices(g)
            fp = np.sum(inputs[idxs], axis=0)
            fingerprints.append(fp)

        return np.vstack(fingerprints)

    def build_weights(self, input_shape):
        """
        Builds the fingerprint. The shape of the fingerprint does not
        necessarily have to be the same as the initial feature vector.
        """
        self.wb.add('weights', shape=(input_shape[0], self.shape))

        output_shape = (input_shape[0], self.shape)

        return output_shape, self.wb


class MaxPoolLayer(object):
    """
    A maxpool layer.
    """

    def __init__(self):
        # self.shape = shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "MaxPoolLayer"

    def forward_pass(self, wb, inputs, graphs):
        """
        For each node and its neighbors, identify the node that has the
        maximum sum of features to be designated as their representative.
        """

        outputs = np.zeros(shape=inputs.shape)
        for g in graphs:
            # g_idxs = graph_indices(g)
            for n, d in g.nodes(data=True):
                sum_feat = np.sum(d['features'])
                # max_idx = g.node[n]['idx']
                idxs = [d['idx']]
                for nbr in g.neighbors(n):
                    nbr_idx = g.node[nbr]['idx']
                    idxs.append(nbr_idx)
                    nbr_sumfeat = np.sum(g.node[nbr]['features'])
                    if nbr_sumfeat > sum_feat:
                        sum_feat = nbr_sumfeat
                        # max_idx = nbr_idx
                outputs[idxs] = sum_feat

        return relu(outputs)

    def build_weights(self, input_shape):
        self.wb.add('weights', shape=input_shape)
        return input_shape, self.wb


class FullyConnectedLayer(object):
    """
    A fully connected layer.
    """

    def __init__(self, shape):
        self.shape = shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "FullyConnectedLayer"

    def forward_pass(self, wb, inputs, graphs):
        return relu(np.dot(inputs, wb['weights']) + wb['bias'])

    def build_weights(self, input_shape):
        self.wb.add('weights', shape=self.shape)
        output_shape = (input_shape[0], self.shape[1])
        self.wb.add('bias', shape=output_shape)

        return output_shape, self.wb


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

    def forward_pass(self, wb, inputs, graphs):
        return inputs * npr.binomial(1, self.p, size=(inputs.shape))

    def build_weights(self, input_shape):
        self.wb.add('weights', shape=input_shape)
        return input_shape, self.wb


class LinearRegressionLayer(object):
    """
    Linear regression layer.
    """
    def __init__(self, shape):
        self.shape = shape
        self.wb = WeightsAndBiases()

    def __repr__(self):
        return "LinearRegressionLayer"

    def forward_pass(self, wb, inputs, graphs):
        return np.dot(inputs, wb['linweights']) + wb['bias']

    def build_weights(self, input_shape):
        self.wb.add('linweights', shape=self.shape)
        output_shape = (input_shape[0], self.shape[1])

        self.wb.add('bias', shape=output_shape)

        return output_shape, self.wb
