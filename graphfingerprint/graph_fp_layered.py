import numpy as np
import custom_funcs as cf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from random import sample, choice
from convnet import GraphInputLayer, GraphConvLayer, FingerprintLayer,\
    LinearRegressionLayer
from wb2 import WeightsAndBiases
from time import time
from autograd import grad

all_nodes = [i for i in range(10)]
lb = LabelBinarizer()
features_dict = {i: lb.fit_transform(all_nodes)[i] for i in all_nodes}

G = cf.make_random_graph(sample(all_nodes, 6), 5, features_dict)
G.edges(data=True)

print('Score of the graph:')
print(cf.score_regressable(G))

n_nodes = [i for i in range(2, len(all_nodes))]
n_graphs = 100

# Make all synthetic graphs
graphs = [cf.make_random_graph(nodes=sample(all_nodes, choice(n_nodes)),
                               n_edges=choice(n_nodes),
                               features_dict=features_dict)
          for i in range(n_graphs)]

input_shape = (1, 10)

layers = [# GraphConvLayer(kernel_shape=(10, 20)),
          # GraphConvLayer(kernel_shape=(20, 20)),
          # GraphConvLayer(kernel_shape=(20, 10)),
          GraphConvLayer(kernel_shape=(10, 10)),
          FingerprintLayer(10),
          LinearRegressionLayer(shape=(10, 1))]
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

wb_all = initialize_network(input_shape, layers, graphs)


def predict(wb_vect, wb_unflattener, inputs, layers, graphs):
    curr_inputs = inputs

    wb_all = wb_unflattener(wb_vect)

    for i, layer in enumerate(layers):
        wb = wb_all['layer{0}_{1}'.format(i, layer)]
        curr_inputs = layer.forward_pass(wb, curr_inputs, graphs)
    return curr_inputs


def train_loss(wb_vect, wb_unflattener, inputs, layers, graphs):
    """
    Training loss is MSE.
    """
    preds = predict(wb_vect, wb_unflattener, inputs, layers, graphs)
    graph_scores = np.array([float(cf.score_regressable(g)) for g in graphs])

    mse = np.sum(np.power(preds - graph_scores, 2)) / len(graphs)
    return mse

gradfunc = grad(train_loss)
wb_vect, wb_unflattener = wb_all.flattened()

# gradfunc(wb_vect, wb_unflattener, inputs, layers, graphs)


def sgd(gradfunc, wb_vect, wb_unflattener, layers, graphs,
        num_iters=200, step_size=0.1, mass=0.9, batch=False, batch_size=10):
    """
    Batch stochastic gradient descent with momentum.
    """

    velocity = np.zeros(len(wb_vect))
    # wb_record = np.zeros(shape=(num_iters, len(wb_vect)))

    training_losses = []
    for i in range(num_iters):
        start = time()

        if batch:
            samp_graphs = sample(graphs, batch_size)
        else:
            samp_graphs = graphs

        samp_inputs = GraphInputLayer(input_shape).forward_pass(samp_graphs)

        print('Epoch: {0}'.format(i))
        print('Computing gradient w.r.t. weights...')
        g = gradfunc(wb_vect, wb_unflattener, samp_inputs, layers, samp_graphs)
        velocity = mass * velocity - (1.0 - mass) * g
        wb_vect += step_size * velocity

        print('Training Loss: ')
        tl = train_loss(wb_vect, wb_unflattener, samp_inputs, layers,
                        samp_graphs)
        print(tl)
        training_losses.append(tl)

        end = time()
        print('Time: {0}'.format(end - start))
        print('')

        step_size = step_size * (1 - step_size)

        print('Step size: {0}'.format(step_size))
    plt.plot(training_losses)
    plt.show()
    # return wb_vect, wb_unflattener

sgd(gradfunc, wb_vect, wb_unflattener, layers, graphs,
    num_iters=1000, step_size=0.0001, batch=True, batch_size=10)

inputs = GraphInputLayer(input_shape).forward_pass(graphs)
# print(predict(wb_vect, wb_unflattener, inputs, layers, graphs))
