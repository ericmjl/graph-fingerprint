from random import sample
from .layers import GraphInputLayer
import math
from .wb2 import WeightsAndBiases

def batch_sample(graphs, input_shape, batch_size=10):
    samp_graphs = sample(graphs, batch_size)
    samp_inputs = GraphInputLayer(input_shape).forward_pass(samp_graphs)

    return samp_graphs, samp_inputs


def y_equals_x(actual_data):
    """
    Returns a 2-tuple of y=x line. Uses the actual_data to set minimum and
    maximum range of y=x.
    """
    minimum = math.floor(min(actual_data))
    maximum = math.ceil(max(actual_data))

    x = [i for i in range(minimum, maximum+1)]
    y = x
    return x, y


def initialize_network(input_shape, graphs, layers_spec):
    """
    Initializes all weights, biases and other parameters to random floats
    between 0 and 1.

    Returns a WeightsAndBiases class that stores all of the parameters
    as well.
    """
    wb_all = WeightsAndBiases()
    curr_shape = input_shape
    for i, layer in enumerate(layers_spec):
        curr_shape, wb = layer.build_weights(curr_shape)
        wb_all['layer{0}_{1}'.format(i, layer)] = wb

    return wb_all

