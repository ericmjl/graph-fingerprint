"""
Author: Eric J. Ma
Date Created: 12 April 2016

Purpose:
What functions can we learn with two convolutional layers?
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import graphfp.custom_funcs as cf
import matplotlib.pyplot as plt
import sys
import seaborn
import os
import pickle as pkl

from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer
from random import sample, choice
from time import time
from autograd import grad
from graphfp.layers import GraphInputLayer, GraphConvLayer, FingerprintLayer,\
    LinearRegressionLayer  # , FullyConnectedLayer
from pyflatten import flatten
from graphfp.optimizers import adam
from graphfp.utils import batch_sample, y_equals_x, initialize_network

seaborn.set_context('poster')


def predict(wb_struct, inputs, nodes_nbrs, graph_idxs):
    """
    Makes predictions by running the forward pass over all of the layers.

    Parameters:
    ===========
    - wb_struct: a dictionary of weights and biases stored for each layer.
    - inputs: the input data matrix. should be one row per graph.
    - graphs: a list of all graphs.

    Internal Variables:
    ===================
    - layers: the list of layer specifications.

    Adding autojit decorator does not speed up code.
    """
    curr_inputs = inputs

    for i, layer in enumerate(layers):
        wb = wb_struct['layer{0}_{1}'.format(i, layer)]
        curr_inputs = layer.forward_pass(wb,
                                         curr_inputs,
                                         nodes_nbrs,
                                         graph_idxs)
    return curr_inputs


def train_loss(wb_vect, unflattener):
    """
    Training loss is MSE.

    We pass in a flattened parameter vector and its unflattener.

    DO NOT JIT this function, it crashes.
    """
    wb_struct = unflattener(wb_vect)
    n_graphs = 100

    samp_graphs, samp_inputs, samp_nodes_nbrs, samp_graph_idxs = batch_sample(
        graphs, input_shape, n_graphs)

    preds = predict(wb_struct, samp_inputs, samp_nodes_nbrs, samp_graph_idxs)
    graph_scores = np.array([float(score_func(g)) for g in samp_graphs]).\
        reshape((len(samp_graphs), 1))

    mse = np.mean(np.power(preds - graph_scores, 2))
    return mse


def callback(wb, i):
    """
    Any function you want to run at each iteration of the optimization.

    Adding autojit decorator does not speed this up.
    """
    start = time()
    wb_vect, wb_unflattener = flatten(wb)
    print('Epoch: {0}'.format(i))

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

        $ python nnet_arch.py cf.score fp_linear 500 10 True

    The arguments are:
    - nnet_arch.py:  the script
    - cf.score:      the scoring function
    - fp_linear:     the network architecture
    - 500:           the number of iterations
    - 10:            the maximum graph size
    - True:          whether or not to make a figure
    """
    start = time()
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

    def array_rep(graphs):
        """
        Takes in a list of graphs, and returns:
        - an arrayed representation of the graphs.
        - a dictionary of graphs to node_idxs
        - a dictionary of graphs to dictionary of idx to node
        - a dictionary of idx to [a node + its neighbors]
        """
        # Do first pass
        curr_idx = 0
        for gid, g in enumerate(graphs):  # gid="graph ID", g=graph
            g.graph['id'] = gid
            g.graph['idxs'] = list()
            for n, d in g.nodes(data=True):
                g.node[n]['idx'] = curr_idx
                g.graph['idxs'].append(curr_idx)
                curr_idx += 1

        graph_arr = np.zeros(shape=(curr_idx, n_feats))
        graph_idxs = defaultdict(dict)
        nodes_nbrs = defaultdict(list)
        # Now update data:
        for g in graphs:
            graph_idxs[g.graph['id']] = g.graph['idxs']
            for n, d in g.nodes(data=True):
                graph_arr[d['idx']] = d['features']
                nodes_nbrs[d['idx']].append(d['idx'])
                for nbr in g.neighbors(n):
                    nodes_nbrs[d['idx']].append(g.node[nbr]['idx'])
        return graph_arr, graph_idxs, nodes_nbrs

    graph_arr, graph_idxs, nodes_nbrs = array_rep(graphs)
    input_shape = (1, 10)

    """
    Set up different simple neural net architectures that can be tested.
    """
    lyr_dict = dict()
    lyr_dict['fp_linear'] = [FingerprintLayer(weights_shape=(n_feats, n_feats),
                                              biases_shape=(1, n_feats)),
                             LinearRegressionLayer(weights_shape=(n_feats, 1),
                                                   biases_shape=(1, 1))]

    lyr_dict['one_conv'] = [GraphConvLayer(weights_shape=(n_feats, n_feats),
                                           biases_shape=(1, n_feats)),
                            FingerprintLayer(weights_shape=(n_feats, n_feats),
                                             biases_shape=(1, n_feats)),
                            LinearRegressionLayer(weights_shape=(n_feats, 1),
                                                  biases_shape=(1, 1))]

    lyr_dict['two_conv'] = [GraphConvLayer(weights_shape=(n_feats, n_feats),
                                           biases_shape=(1, n_feats)),
                            GraphConvLayer(weights_shape=(n_feats, n_feats),
                                           biases_shape=(1, n_feats)),
                            FingerprintLayer(weights_shape=(n_feats, n_feats),
                                             biases_shape=(1, n_feats)),
                            LinearRegressionLayer(weights_shape=(n_feats, 1),
                                                  biases_shape=(1, 1))]

    lyr_dict['1conv_expd'] = [GraphConvLayer(weights_shape=(n_feats,
                                                            2*n_feats),
                                             biases_shape=(1, n_feats)),
                              FingerprintLayer(weights_shape=(2*n_feats,
                                                              n_feats),
                                               biases_shape=(1, n_feats)),
                              LinearRegressionLayer(weights_shape=(n_feats, 1),
                                                    biases_shape=(1, 1))]

    # lyr_dict['full_connect'] = [GraphConvLayer((n_feats, n_feats)),
    #                             FingerprintLayer(n_feats),
    #                             FullyConnectedLayer((n_feats, n_feats)),
    #                             LinearRegressionLayer((n_feats, 1))]

    layers = lyr_dict[arch]

    gradfunc = grad(train_loss)
    training_losses = []
    wb_all = initialize_network(layers)

    """Train on testing data."""
    wb_vect, wb_unflattener = adam(gradfunc, wb_all, callback=callback,
                                   num_iters=num_iters, step_size=0.01)

    """Write the training losses."""
    def make_training_loss_figure():
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_ylabel('training error')
        ax.set_xlabel('iteration')
        ax.plot(training_losses)
        plt.subplots_adjust(bottom=0.2, left=0.25, top=0.9, right=0.92)
        plt.savefig('figures/{3}-{0}-{1}_iters-{2}_feats-training_loss.pdf'
                    .format(func_name, num_iters, n_feats, arch))

    # inputs, nodes_nbrs, graph_idxs = GraphInputLayer(input_shape).\
    #     forward_pass(graphs)

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

    new_inputs, new_nodes_nbrs, new_graph_idxs = GraphInputLayer(input_shape).\
        forward_pass(new_graphs)

    preds = predict(wb_new, new_inputs, new_nodes_nbrs, new_graph_idxs)
    actual = np.array([score_func(g) for g in new_graphs]).reshape(
        (len(new_graphs), 1))

    """Make a scatterplot of the actual vs. predicted values on new data."""
    def make_scatterplot_figure():
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(actual, preds, color='red', label='predictions')
        ax.set_xlabel('actual')
        ax.set_ylabel('predictions')
        ax.plot(*y_equals_x(actual), color='red')  # this is the y=x line
        plt.subplots_adjust(bottom=0.2, left=0.25, top=0.9, right=0.92)
        plt.savefig('figures/{3}-{0}-{1}_iters-{2}_feats-preds_vs_actual.pdf'
                    .format(func_name, num_iters, n_feats, arch))

    def make_dir(dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    if make_plots:
        make_dir('figures')
        make_scatterplot_figure()
        make_training_loss_figure()

    # Save the weights and baises to disk.
    def save_wb():
        make_dir('wbs')
        with open('wbs/{3}-{0}-{1}_iters-{2}_wb.pkl'
                  .format(func_name, num_iters, n_feats, arch), 'wb') as f:
            pkl.dump(wb_new, f)

    end = time()
    print('\n Total Time:')
    print(end - start)

    print('\n Saving weights and biases to disk')
    save_wb()
