import numpy as np
import custom_funcs as cf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from random import sample, choice
from convnet import GraphInputLayer, GraphConvLayer, FingerprintLayer,\
    LinearRegressionLayer, FullyConnectedLayer, MaxPoolLayer
from wb2 import WeightsAndBiases
from time import time
from autograd import grad
from autograd.util import check_grads
from flatten import flatten
# from optimizers import sgd

n_feats = 6
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

layers = [# GraphConvLayer(kernel_shape=(n_feats, 2*n_feats)),
          # GraphConvLayer(kernel_shape=(20, 20)),
          # GraphConvLayer(kernel_shape=(2*n_feats, n_feats)),
          GraphConvLayer(kernel_shape=(n_feats, n_feats)),
          # MaxPoolLayer(),
          FingerprintLayer(n_feats),
          # FullyConnectedLayer((10, 10)),
          # FullyConnectedLayer((10, 1)),
          LinearRegressionLayer(shape=(n_feats, 1)),
          ]
print(layers)


def initialize_network(input_shape, layers, graphs):
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


def predict(wb_struct, inputs, layers, graphs):
    curr_inputs = inputs

    for i, layer in enumerate(layers):
        wb = wb_struct['layer{0}_{1}'.format(i, layer)]
        curr_inputs = layer.forward_pass(wb, curr_inputs, graphs)
    return curr_inputs

# old signature below:
# def train_loss(wb_vect, i, inputs, layers, graphs):
# new signature


def train_loss(wb_vect, unflattener, inputs, layers, graphs):
    """
    Training loss is MSE.

    We pass in a flattened parameter vector.
    """
    wb_struct = unflattener(wb_vect)
    preds = predict(wb_struct, inputs, layers, graphs)
    graph_scores = np.array([float(score_func(g)) for g in graphs]).reshape(
        (len(graphs), 1))

    mse = np.mean(np.power(preds - graph_scores, 2))
    return mse

gradfunc = grad(train_loss)

# Check gradients
# inputs = GraphInputLayer(input_shape).forward_pass(graphs)
# oatl = lambda w: train_loss(w, wb_unflattener, inputs, layers, graphs)
# check_grads(oatl, wb_vect)


def callback(wb, inputs, layers, graphs, i):
    start = time()
    wb_vect, wb_unflattener = flatten(wb)
    print('Epoch: {0}'.format(i))
    # print('Computing gradient w.r.t. weights...')

    print('Training Loss: ')
    tl = train_loss(wb_vect, wb_unflattener, inputs, layers, graphs)
    print(tl)
    # training_losses.append(tl)

    end = time()
    print('Time: {0}'.format(end - start))
    print('')
    return tl


def sgd(gradfunc, wb, layers, graphs, callback=None,
        num_iters=200, step_size=0.1, mass=0.9, batch=False,
        batch_size=10, adaptive=False):
    """
    Batch stochastic gradient descent with momentum.

    Todo:
    - Refactor to make this follow the SGD signature and code in the autograd
      examples.
    """
    wb_vect, wb_unflattener = flatten(wb)
    velocity = np.zeros(len(wb_vect))

    training_losses = []
    for i in range(num_iters):
        # start = time()

        if batch:
            samp_graphs = sample(graphs, batch_size)
        else:
            samp_graphs = graphs
        samp_inputs = GraphInputLayer(input_shape).forward_pass(samp_graphs)

        g = gradfunc(wb_vect, wb_unflattener, samp_inputs, layers, samp_graphs)
        velocity = mass * velocity - (1.0 - mass) * g
        wb_vect += step_size * velocity
        if adaptive:
            step_size = step_size * (1 - step_size)

        # Diagnostic statements.
        wb = wb_unflattener(wb_vect)
        tl = callback(wb, samp_inputs, layers, samp_graphs, i)
        training_losses.append(tl)
        print('Step size: {0}'.format(step_size))
    plt.plot(training_losses)
    plt.show()

    return wb_vect, wb_unflattener


wb_all = initialize_network(input_shape, layers, graphs)

wb_vect, wb_unflattener = sgd(gradfunc, wb_all, layers, graphs,
                              callback=callback, num_iters=2000, step_size=0.1,
                              batch=True, batch_size=20, adaptive=True)


inputs = GraphInputLayer(input_shape).forward_pass(graphs)
print('Final training loss:')
print(train_loss(wb_vect, wb_unflattener, inputs, layers, graphs))

wb_new = wb_unflattener(wb_vect)

print('Predictions on Training Data:')
preds = predict(wb_new, inputs, layers, graphs)
print(preds)
print('Actual:')
scores = [score_func(g) for g in graphs]
print(scores)
print(wb_unflattener(wb_vect))

n_new_graphs = 10
new_graphs = [cf.make_random_graph(nodes=sample(all_nodes, choice(n_nodes)),
                                   n_edges=choice(n_nodes),
                                   features_dict=features_dict)
              for i in range(n_new_graphs)]

new_inputs = GraphInputLayer(input_shape).forward_pass(new_graphs)

preds = predict(wb_new, new_inputs, layers, new_graphs)
actual = np.array([score_func(g) for g in new_graphs]).reshape(
    (len(new_graphs), 1))

print('Predictions:')
print(preds)

print('Actual:')
print(actual)

print('Absolute Error:')
print(np.abs(preds - actual))
