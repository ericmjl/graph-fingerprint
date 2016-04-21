"""
Author: Eric J. Ma
Date Created: 12 April 2016

Purpose:
What functions can we learn with two convolutional layers?
"""

import numpy as np
import graphfp.custom_funcs as cf
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import LabelBinarizer
from random import sample, choice
from time import time
from autograd import grad
from graphfp.layers import GraphInputLayer, GraphConvLayer, FingerprintLayer,\
    LinearRegressionLayer
from graphfp.wb2 import WeightsAndBiases
from graphfp.flatten import flatten
from graphfp.optimizers import sgd
from graphfp.utils import batch_sample, y_equals_x, initialize_network
# from autograd.util import check_grads


def predict(wb_struct, inputs, graphs):
    """
    Makes predictions by running the forward pass over all of the layers.

    Parameters:
    ===========
    - wb_struct: a dictionary of weights and biases stored for each layer.
    - inputs: the input data matrix. should be one row per graph.
    - graphs: a list of all graphs.
    """
    curr_inputs = inputs

    for i, layer in enumerate(layers):
        wb = wb_struct['layer{0}_{1}'.format(i, layer)]
        curr_inputs = layer.forward_pass(wb, curr_inputs, graphs)
    return curr_inputs


def train_loss(wb_vect, unflattener, batch=True, batch_size=10):
    """
    Training loss is MSE.

    We pass in a flattened parameter vector and its unflattener.
    """
    wb_struct = unflattener(wb_vect)

    if batch:
        batch_size = batch_size
    else:
        batch_size = len(graphs)

    samp_graphs, samp_inputs = batch_sample(graphs, input_shape, batch_size)

    preds = predict(wb_struct, samp_inputs, samp_graphs)
    graph_scores = np.array([float(score_func(g)) for g in samp_graphs]).\
        reshape((len(samp_graphs), 1))

    mse = np.mean(np.power(preds - graph_scores, 2))
    return mse


def callback(wb, i):
    """
    Any function you want to run at each iteration of the optimization.
    """
    start = time()
    wb_vect, wb_unflattener = flatten(wb)
    print('Epoch: {0}'.format(i))
    # print('Computing gradient w.r.t. weights...')

    print('Training Loss: ')

    tl = train_loss(wb_vect, wb_unflattener)
    print(tl)

    end = time()
    print('Time: {0}'.format(end - start))
    print('')

    training_losses.append(tl)

if __name__ == '__main__':

    """
    Set up hyperparameters.

    The signature at the command line will look like:

        $ python nnet_arch.py cf.score fp_linear 500 10 False
    """
    func_name = sys.argv[1]
    score_func = eval(sys.argv[1])
    arch = sys.argv[2]
    num_iters = int(sys.argv[3])
    n_feats = int(sys.argv[4])
    make_plots = eval(sys.argv[5])

    """Initialize graphs."""
    all_nodes = [i for i in range(n_feats)]
    lb = LabelBinarizer()
    features_dict = {i: lb.fit_transform(all_nodes)[i] for i in all_nodes}

    n_nodes = [i for i in range(2, len(all_nodes))]  # choose from here the
        # num of nodes to add.
    n_graphs = 1000  # the number of synthetic graphs to make as training data.

    # Make all synthetic graphs
    graphs = [cf.make_random_graph(nodes=sample(all_nodes, choice(n_nodes)),
                                   n_edges=choice(n_nodes),
                                   features_dict=features_dict)
              for i in range(n_graphs)]

    input_shape = (1, 10)

    """Set up different simple neural net architectures that can be tested."""
    lyr_dict = dict()
    lyr_dict['fp_linear'] = [FingerprintLayer(n_feats),
                             LinearRegressionLayer(shape=(n_feats, 1))]

    lyr_dict['one_conv'] = [GraphConvLayer((n_feats, n_feats)),
                            FingerprintLayer(n_feats),
                            LinearRegressionLayer(shape=(n_feats, 1))]

    lyr_dict['two_conv'] = [GraphConvLayer((n_feats, n_feats)),
                            GraphConvLayer((n_feats, n_feats)),
                            FingerprintLayer(n_feats),
                            LinearRegressionLayer(shape=(n_feats, 1))]

    layers = lyr_dict[arch]

    gradfunc = grad(train_loss)
    training_losses = []
    wb_all = initialize_network(input_shape, graphs, layers)

    """Train on testing data."""
    wb_vect, wb_unflattener = sgd(gradfunc, wb_all, layers, graphs,
                                  callback=callback, num_iters=num_iters,
                                  step_size=0.1, adaptive=True)

    """Print the training losses."""
    def make_training_loss_figure():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.plot(training_losses)
        plt.savefig('figures/{3}-{0}-{1}_iters-{2}_feats-training_loss.pdf'
                    .format(func_name, num_iters, n_feats, arch))

    if make_plots:
        make_training_loss_figure()

    inputs = GraphInputLayer(input_shape).forward_pass(graphs)
    print('Final training loss:')
    print(train_loss(wb_vect, wb_unflattener))

    wb_new = wb_unflattener(wb_vect)

    """Make predictions on new graphs"""
    n_new_graphs = 100
    new_graphs = [cf.make_random_graph(nodes=sample(all_nodes,
                                                    choice(n_nodes)),
                                       n_edges=choice(n_nodes),
                                       features_dict=features_dict)
                  for i in range(n_new_graphs)]

    new_inputs = GraphInputLayer(input_shape).forward_pass(new_graphs)

    preds = predict(wb_new, new_inputs, new_graphs)
    actual = np.array([score_func(g) for g in new_graphs]).reshape(
        (len(new_graphs), 1))

    """Make a scatterplot of the actual vs. predicted values on new data."""
    def make_scatterplot_figure():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(actual, preds, color='red', label='predictions')
        ax.set_xlabel('actual')
        ax.set_ylabel('predictions')
        ax.plot(*y_equals_x(actual), color='red')  # this is the y=x line
        ax.legend()
        plt.savefig('figures/{3}-{0}-{1}_iters-{2}_feats-preds_vs_actual.pdf'
                    .format(func_name, num_iters, n_feats, arch))
        # plt.show()

    if make_plots:
        make_scatterplot_figure()
