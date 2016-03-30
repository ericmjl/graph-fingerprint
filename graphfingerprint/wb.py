from flatten import flatten
import autograd.numpy.random as npr


class WeightsAndBiases(dict):
    """
    A class that stores the WeightsAndBiases for convolution.
    """
    def __init__(self, n_layers, shapes):
        super(WeightsAndBiases, self).__init__()
        self.initialize_layers(n_layers, shapes)
        self.vect, self.unflattener = self.flattened()

    def __iter__():
        pass

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

            if i == 0:
            # TODO: Change initialized values to center on 0.
                self[i]['self_weights'] = npr.random((shapes[i], shapes[i]))
                self[i]['nbr_weights'] = npr.random((shapes[i], shapes[i]))
                self[i]['biases'] = npr.random((1, shapes[i]))
            else:
                self[i]['self_weights'] = npr.random((shapes[i-1], shapes[i]))
                self[i]['nbr_weights'] = npr.random((shapes[i-1], shapes[i]))
                self[i]['biases'] = npr.random((1, shapes[i]))

        self[n_layers]['linweights'] = npr.random((shapes[i], 1))

    def flattened(self):
        flattened_weights, unflattener = flatten(self)

        return flattened_weights, unflattener
