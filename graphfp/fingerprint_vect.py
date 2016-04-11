"""
An implementation of the graph fingerprinting using vectorization in numpy.
"""
import networkx as nx
import autograd.numpy as np


class GraphFingerprint(nx.Graph):
    """
    docstring for GraphFingerprint
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

        assert len(feat_lengths) == 1, \
            "feature vectors are not of the same length."

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
                    layers[i].node[node]['features'] = np.zeros(shapes[i])

        return layers

    def node_activations(self):
        """
        Grabs out the node activations from the graph layer as a dictionary.
        Dictionary keys are the node, and values are the feature vector.
        """
        features = dict()
        for n, d in self.graph.nodes(data=True):
            features[n] = d['features']
        return features

    # def nbr_activations(self):
    #     """
    #     Grabs out the neighbor activations from the graph layer, sums them up,
    #     and returns them as a stacked numpy array.
    #     """
    #     features = []
    #     G = self.layers[layer]
    #     for n, d in G.nodes(data=True):
    #         neighbors = np.zeros(shape=d['features'].shape)
    #         for nbr in G.neighbors(n):
    #             neighbors = neighbors + G.node[nbr]['features']
    #         features.append(neighbors)
    #     return np.stack(features)

    def set_matrix_indexes(self, indices_dict):
        """
        The fingerprint objects are used with a larger vector.
        """
