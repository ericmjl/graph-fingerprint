from pyflatten import flatten
import autograd.numpy.random as npr


class WeightsAndBiases(dict):
    """
    A class that stores the WeightsAndBiases for convolution.
    """
    def __init__(self):
        # super(WeightsAndBiases, self).__init__()
        # self.initialize_layers(n_layers, shapes)
        # self.vect, self.unflattener = self.flattened()
        pass

    def flattened(self):
        """
        Deprecated. Don't bother with this.
        """
        flattened_weights, unflattener = flatten(self)
        self.vect = flattened_weights
        self.unflattener = unflattener
        return flattened_weights, unflattener

    def add(self, name, shape):
        """
        Add a randomly initialized set of weights/biases to the WB class. It
        is initialized with mean 0 and variance 0.1

        Parameters:
        ===========
        - name: (string) self_weights, nbr_weights, biases, or some other name.
        - shape: (tuple) the dimensions of the layer.
        """
        self[name] = npr.normal(0, 0.001, shape)
        # self[name] = np.ones(shape)
