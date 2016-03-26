import autograd.numpy as np


class WeightsAndBiases(dict):
    """A class that stores the WeightsAndBiases for convolution."""
    def __init__(self, n_layers, shapes):
        super(WeightsAndBiases, self).__init__()
        self.layers = self.initialize_layers(n_layers, shapes)
        self.shapes = shapes

    def initialize_layers(self, n_layers, shapes):
        """
        Initializes the layers with random matrices of the correct shapes.

        Parameters:
        ===========
        n_layers: (int) the number of layers present.
        shapes: (dict of 2-tuples) the shapes of each of the weights, in the
                form of (n_rows, n_columns).
        """

        for i in range(n_layers+1):
            self[i] = dict()
            self[i]['self_weights'] = np.random.random((shapes[i], shapes[i]))
            self[i]['nbr_weights'] = np.random.random((shapes[i], shapes[i]))
            self[i]['biases'] = np.random.random((1, shapes[i]))
