"""
A class to represent a Graph Convolutional Neural Network.
"""

import networkx as nx
from collections import OrderedDict
from autograd import numpy as np
from autograd.scipy.misc import logsumexp


def softmax(X, axis=0):
    """
    The softmax function normalizes everything to between 0 and 1.
    """
    return np.exp(X - logsumexp(X, axis=axis, keepdims=True))


class GraphConvNN(object):
    """
    GraphConvNN: A class to represent a Graph Convolutional Neural Network.

    The data model is currently structured as such, and can be changed:
    - self.layers: a dictionary where keys = the layer index, and values = the
      graph representation at that layer. The 0-th layer is the input data
      layer.
    - self.fingerprint: a numpy array of specified shape.

    The math that occurs on the graph is as such:
    - each node has a feature vector, multiplied by a "self" weight matrix
      stored in its attributes, giving the activation for itself. The "self"
      weight matrix is stored in the 'weights_self' node attribute.
    - each node's neighbors are summed and multiplied by a "neighbors" weight
      matrix stored in its attributes, giving the activation with its
      neighbors. The "neighbors" weight matrix is stored in the
      "weights_neighbords" node attribute.
    - each node has a bias vector that is applied to the sum of the "self" and
      "neighbor" activations. The "bias" vector is stored in the "bias" node
      attribute.
    """
    def __init__(self, graph, n_layers):
        super(GraphConvNN, self).__init__()
        self.layers = self.set_layers(n_layers, graph)
        self.fingerprint = None

    def check_graph(self, graph):
        """
        Performs validation checks on the structure of the graph data.

        If validation checks pass, returns the graph back.
        """
        assert isinstance(graph, nx.Graph)

        feat_lengths = set()
        for n, d in graph.nodes(data=True):
            assert 'features' in d.keys(), "'features' atribute missing."
            feat_lengths.add(len(d['features']))

        assert len(feat_lengths) == 1, "feature vectors not same length."

    def set_layers(self, n_layers, graph):
        """
        Sets the self.layers attribute to a dictionary, where each layer
        represents one convolution operation performed on the previous layer.
        """

        # Defensive programming checks.
        assert isinstance(n_layers, int)
        self.check_graph(graph)

        # Add the graph to the layers.
        # We use an OrderedDict to ensure that the layers are added in
        # sequentially.
        layers = OrderedDict()
        for i in range(n_layers + 1):
            layers[i] = graph.copy()

        return layers

    def initialize(self):
        """
        Initializes the weights and biases on the convolutional neural network.
        """
        def initialize_node_weights(n, graph, n_dims):
            """
            Helper function to initialize node weights.
            """
            graph.node[n]['weights_self'] = np.random.random((n_dims, n_dims))
            graph.node[n]['weights_neighbors'] = np.random.random((n_dims,
                                                                   n_dims))

        def initialize_node_biases(n, graph, n_dims):
            """
            Helper function to initialize node biases.
            """
            graph.node[n]['bias'] = np.random.random((1, n_dims))

        # Initialize the weights on every other layer.
        for layer, graph in self.layers.items():
            for n, d in graph.nodes(data=True):
                n_dims = len(d['features'])
                initialize_node_weights(n, graph, n_dims)
                initialize_node_biases(n, graph, n_dims)

    def compute_node_activations(self):
        """
        This is the code for computing node activations. The computations will
        take place on each layer, starting with layer 1 that will be learned
        from layer 0, layer 2 that will be learned from layer 1 etc.
        """
        for layer, graph in self.layers.items():
            if layer != max(self.layers.keys()):
                for n, d in graph.nodes(data=True):
                    # Compute self activation.
                    self_a = np.dot(d['features'], d['weights_self'])

                    # Compute neighbor activation.
                    neighbor_a = np.zeros(len(d['features']))
                    for nbr in graph.neighbors(n):
                        neighbor_a = neighbor_a + graph.node[nbr]['features']
                    neighbor_a = np.dot(neighbor_a, d['weights_neighbors'])

                    # Compute sum activation
                    total_a = self_a + neighbor_a + d['bias']
                    # total_a = softmax(total_a)

                    # Assign the activation to the graph one layer above.
                    self.layers[layer + 1].node[n]['features'] = total_a

    def compute_fingerprint(self):
        """
        Sums the vectors on each node on the top layer to produce the
        fingerprint.

        Thoughts for the future:
        - allow the input of a matrix of weights, so that the fingerprint can
          be expanded or shrunk in length.
        """
        top_layer = max(self.layers.keys())
        graph = self.layers[top_layer]
        n_dims = len(graph.nodes(data=True)[0][1]['features'])
        fingerprint = np.zeros((1, n_dims))
        for n, d in graph.nodes(data=True):
            fingerprint = fingerprint + d['features']

        # fingerprint = softmax(fingerprint)

        return fingerprint
