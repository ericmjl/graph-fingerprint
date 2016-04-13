"""
Author: Eric J. Ma
Date Created: 12 April 2016

Purpose:
Can we learn a sine transformation on top of a convolution?
"""

import numpy as np
import graphfp.custom_funcs as cf
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import LabelBinarizer
from random import sample, choice
from time import time
from autograd import grad
from graphfp.layers import GraphInputLayer, GraphConvLayer, FingerprintLayer,\
    LinearRegressionLayer
from graphfp.wb2 import WeightsAndBiases
from graphfp.flatten import flatten
from graphfp.optimizers import sgd
from graphfp.utils import batch_sample
# from autograd.util import check_grads

n_feats = 30
all_nodes = [i for i in range(n_feats)]
lb = LabelBinarizer()
features_dict = {i: lb.fit_transform(all_nodes)[i] for i in all_nodes}

G = cf.make_random_graph(sample(all_nodes, n_feats-1), n_feats-2,
                         features_dict)
print(G.nodes(data=True))

score_func = cf.score_sine

print('Score of the graph:')
print(score_func(G))

n_nodes = [i for i in range(2, len(all_nodes))]
n_graphs = 1000

# Make all synthetic graphs
graphs = [cf.make_random_graph(nodes=sample(all_nodes, choice(n_nodes)),
                               n_edges=choice(n_nodes),
                               features_dict=features_dict)
          for i in range(n_graphs)]

input_shape = (1, 10)

layers = [GraphConvLayer(kernel_shape=(n_feats, n_feats)),
          FingerprintLayer(n_feats),
          LinearRegressionLayer(shape=(n_feats, 1))]


def initialize_network(input_shape, graphs):
    """
    Initializes all weights, biases and other parameters to random floats
    between 0 and 1.

    Returns a WeightsAndBiases class that stores all of the parameters
    as well.
    """
    wb_all = WeightsAndBiases()
    curr_shape = input_shape
    for i, layer in enumerate(layers):
        curr_shape, wb = layer.build_weights(curr_shape)
        wb_all['layer{0}_{1}'.format(i, layer)] = wb

    return wb_all


def predict(wb_struct, inputs, graphs):
    """
    Makes predictions by running the forward pass over all of the layers.
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

gradfunc = grad(train_loss)

training_losses = []
fig_tl = plt.figure()


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

    """Make plot of training losses"""
    ax = fig.add_subplot(111)
    ax.plot(training_losses)
    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('training loss')



wb_all = initialize_network(input_shape, graphs)

"""Train on testing data."""
wb_vect, wb_unflattener = sgd(gradfunc, wb_all, layers, graphs,
                              callback=callback, num_iters=1000, step_size=0.1,
                              adaptive=True)



inputs = GraphInputLayer(input_shape).forward_pass(graphs)
print('Final training loss:')
print(train_loss(wb_vect, wb_unflattener))

wb_new = wb_unflattener(wb_vect)

"""Make predictions on new graphs"""
n_new_graphs = 100
new_graphs = [cf.make_random_graph(nodes=sample(all_nodes, choice(n_nodes)),
                                   n_edges=choice(n_nodes),
                                   features_dict=features_dict)
              for i in range(n_new_graphs)]

new_inputs = GraphInputLayer(input_shape).forward_pass(new_graphs)


preds = predict(wb_new, new_inputs, new_graphs)
actual = np.array([score_func(g) for g in new_graphs]).reshape(
    (len(new_graphs), 1))

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

"""Make a scatterplot of the actual vs. predicted values on new data."""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(actual, preds, color='red', label='predictions')
ax.set_xlabel('actual')
ax.set_ylabel('predictions')
ax.plot(*y_equals_x(actual), color='red')  # this is the y=x line
# plt.scatter(np.arcsin(actual), actual, color='blue', label='actual')
ax.legend()
plt.show()
