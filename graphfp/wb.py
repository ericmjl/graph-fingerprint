from pyflatten import flatten
import autograd.numpy.random as npr


class WeightsAndBiases(dict):
    """
    A class that stores the WeightsAndBiases for convolution.
    """
    def __init__(self, n_layers, shapes):
        super(WeightsAndBiases, self).__init__()
        self.initialize_layers(n_layers, shapes)
        self.vect, self.unflattener = self.flattened()

    def flattened(self):
        flattened_weights, unflattener = flatten(self)

        return flattened_weights, unflattener

    def add(self, name, shape):
        """
        Add a layer to the WB class.

        Parameters:
        ===========
        - name: (string) self_weights, nbr_weights, biases, or some other name.
        - shape: (tuple) the dimensions of the layer.
        """
        self[name] = npr.normal(0, 1, shape)

    def initialize_layers(self, n_layers, shapes):
        """
        Initializes the layers with random matrices of the correct shapes.

        Parameters:
        ===========
        n_layers: (int) the number of layers present.
        shapes:   (dict) the shapes of the weights at each layer, in the form
                  of n_columns.
        """

        for i in range(n_layers+1):
            self[i] = dict()

            if i != n_layers:
            # TODO: Change initialized values to center on 0.
                self[i]['self_weights'] = npr.normal(0, 0.1, (shapes[i], shapes[i+1]))
                self[i]['nbr_weights'] = npr.normal(0, 0.1, (shapes[i], shapes[i+1]))
                self[i]['biases'] = npr.normal(0, 0.1, (1, shapes[i+1]))
            else:
                self[i]['self_weights'] = npr.normal(0, 0.1, (shapes[i], shapes[i]))
                self[i]['nbr_weights'] = npr.normal(0, 0.1, (shapes[i], shapes[i]))
                self[i]['biases'] = npr.normal(0, 0.1, (1, shapes[i]))

        self[n_layers]['linweights'] = npr.normal(0, 0.1, (shapes[i], 1))

