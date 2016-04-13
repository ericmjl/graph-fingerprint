from random import sample
from .layers import GraphInputLayer


def batch_sample(graphs, input_shape, batch_size=10):
    samp_graphs = sample(graphs, batch_size)
    samp_inputs = GraphInputLayer(input_shape).forward_pass(samp_graphs)

    return samp_graphs, samp_inputs
