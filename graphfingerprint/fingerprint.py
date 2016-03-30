"""
A class to compute a fingerprint from a graph using graph convolutions.
"""

import networkx as nx
from autograd import numpy as np
from autograd.scipy.misc import logsumexp


def softmax(X, axis=0):
    """
    The softmax function normalizes everything to between 0 and 1.
    """
    return np.exp(X - logsumexp(X, axis=axis, keepdims=True))

def relu(X, axis=0):
    return X * (X > 0)


class GraphFingerprint(object):
    """
    GraphFingerprint: A class to perform convolutions on graphs.

    The data model is currently structured as such, and can be changed:
    - self.layers: a dictionary where keys = the layer index, and values = the
      graph representation at that layer. The 0-th layer is the input data
      layer.

    The math that occurs on the graph is as such:
    - each node has a feature vector, multiplied by a "self" weight matrix
      associated with the layer, giving the activation for itself. The "self"
      weight matrix is stored in the 'weights_self' node attribute.
    - each node's neighbors are summed and multiplied by a "neighbors" weight
      matrix associated with the layer, giving the activation with its
      neighbors. The "neighbors" weight matrix is stored in the
      "weights_neighbords" node attribute.
    - each node has a bias vector that is applied to the sum of the "self" and
      "neighbor" activations. The "bias" vector is stored in the "bias" node
      attribute.
    """
    def __init__(self, graph, n_layers, shapes):
        super(GraphFingerprint, self).__init__()
        self.layers = self.set_layers(n_layers, graph, shapes)

    def check_graph(self, graph):
        """
        Performs validation checks on the structure of the graph data.
        """
        assert isinstance(graph, nx.Graph)

        feat_lengths = set()
        for n, d in graph.nodes(data=True):
            assert 'features' in d.keys(), "'features' atribute missing."
            feat_lengths.add(len(d['features']))

        assert len(feat_lengths) == 1, "feature vectors not same length."

    def set_layers(self, n_layers, graph, shapes):
        """
        Sets the self.layers attribute to a dictionary, where each layer
        represents one convolution operation performed on the previous layer.

        Parameters:
        ===========
        - n_layers: (int) the number of layers
        - graph:    (nx.Graph) the graph on which the convolutions are to be
                    performed.
        - shapes:   (tuple) the feature lengths at each layer, including the
                    input layer.
        """
        # Defensive programming checks.
        assert isinstance(n_layers, int)
        self.check_graph(graph)

        # Add the graph to the layers.
        layers = dict()
        for i in range(n_layers + 1):
            layers[i] = graph.copy()
            if i != 0:
                for node in layers[i].nodes():
                    layers[i].node[node]['features'] = np.zeros((1, shapes[i]))

        return layers

    def compute_node_activations(self, wb_vect, wb_unflattener):
        """
        This is the code for computing node activations. The computations will
        take place on each layer, starting with layer 1 that will be learned
        from layer 0, layer 2 that will be learned from layer 1 etc.

        Parameters:
        ===========
        wb_vect:        (vector) the weights and biases for each layer.
        wb_unflattener: (list) the list of unflattener functions for wb_vect.
        """
        wb = wb_unflattener(wb_vect)
        layers = sorted([i for i in self.layers.keys()])
        for layer in layers:
            if layer != max(self.layers.keys()):
                graph = self.layers[layer]
                for n, d in graph.nodes(data=True):
                    # Compute self activation.
                    self_a = np.dot(d['features'], wb[layer]['self_weights'])

                    # Compute neighbor activation.
                    neighbor_a = np.zeros((1, d['features'].shape[0]))
                    for nbr in graph.neighbors(n):
                        # print(n, layer, nbr)
                        neighbor_a = neighbor_a + graph.node[nbr]['features']
                        neighbor_a = np.dot(neighbor_a,
                                            wb[layer]['nbr_weights'])

                    # Compute sum activation
                    total_a = self_a + neighbor_a + wb[layer]['biases']
                    # total_a = relu(total_a, axis=1)
                    total_a = softmax(total_a, axis=1)

                    # Assign the activation to the graph one layer above.
                    self.layers[layer + 1].node[n]['features'] = total_a

    def compute_fingerprint(self, wb_vect, wb_unflattener):
        """
        Sums the vectors on each node on the top layer to produce the
        fingerprint. This is the simplest implementation.

        Thoughts for the future:
        - allow the input of a matrix of weights, so that the fingerprint can
          be expanded or shrunk in length.
        """
        self.compute_node_activations(wb_vect, wb_unflattener)

        top_layer = max(self.layers.keys())
        graph = self.layers[top_layer]
        n_dims = len(graph.nodes(data=True)[0][1]['features'])

        # Summing happens here.
        fingerprint = np.zeros((1, n_dims))
        for n, d in graph.nodes(data=True):
            fingerprint = fingerprint + d['features']

        fingerprint = softmax(fingerprint, axis=1)

        return fingerprint
