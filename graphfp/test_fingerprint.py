# from graphfp import GraphFingerprint
# from wb import WeightsAndBiases
# from itertools import combinations
# from random import sample, choice
# import autograd.numpy as np
# import networkx as nx

# # Initialize shapes of weights & biases class.
# shapes = dict()
# shapes[0] = 10
# shapes[1] = 10
# shapes[2] = 10
# wb = WeightsAndBiases(2, shapes)

# # Generate n synthetic graphs that have a random configuration of nodes which
# # have fixed feature vectors.
# # - nodes are 'A' through 'G'
# # - select random set of nodes to add to the graph.
# # - choose a number of edges to add, and add them in randomly to the graph.


# def rnd():
#     return np.random.binomial(1, 0.2, size=10)

# all_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


# def make_nodes_with_features():
#     features = dict()
#     for letter in all_letters:
#         features[letter] = rnd()

#     return features

# node_features = make_nodes_with_features()


# def make_synthetic_graphs(num_graphs, features):
#     num_nodes = [i for i in range(2, len(all_letters) + 1)]

#     # Make the synthetic graphs.
#     syngraphs = []  # the synthetic graphs
#     for i in range(num_graphs):
#         # add in nodes
#         n_nodes = choice(num_nodes)
#         letters = sample(all_letters, n_nodes)
#         G = nx.Graph()
#         for letter in letters:
#             G.add_node(letter, features=features[letter])

#         # add in edges
#         n_nodes = len(G.nodes())
#         num_edges = choice(range(1, int(n_nodes**2 / 2 - n_nodes / 2 + 1)))
#         edges = sample([i for i in combinations(G.nodes(), 2)], num_edges)
#         for u, v in edges:
#             G.add_edge(u, v)
#         syngraphs.append(G)
#     return syngraphs

# syngraphs = make_synthetic_graphs(10, node_features)


# def test_same_graphs_have_same_fingerprints():
#     """
#     Given two graphs that are of the same structure, and a set of weights &
#     biases, check that the resultant fingerprints are equal.
#     """
#     graph = choice(syngraphs)
#     gfp1 = GraphFingerprint(graph, 2, shapes)
#     gfp2 = GraphFingerprint(graph, 2, shapes)

#     fp1 = gfp1.compute_fingerprint(wb.vect, wb.unflattener)
#     fp2 = gfp2.compute_fingerprint(wb.vect, wb.unflattener)

#     assert np.array_equal(fp1, fp2)

# def test_different_layers_have_different_fingerprints():
#     """
#     Given a graph, check that for a given node, its fingerprint in one layer
#     is not the same as the fingerprint in another layer..
#     """
#     graph = choice(syngraphs)
#     gfp = GraphFingerprint(graph, 2, shapes)
#     gfp.compute_node_activations(wb.vect, wb.unflattener)

#     node = choice(gfp.layers[0].nodes())
#     act_l0 = gfp.layers[0].node[node]['features']
#     act_l1 = gfp.layers[1].node[node]['features']
#     act_l2 = gfp.layers[2].node[node]['features']

#     assert not np.array_equal(act_l0, act_l1)
#     assert not np.array_equal(act_l1, act_l2)


# def test_same_nodes_diff_edges_have_diff_fingerprints():
#     """
#     Given two graphs that have the same nodes connected differently, and a set
#     of weights and biases, check to make sure that their fingerprints turn out
#     being different.
#     """
#     nodes = sample(all_letters, 4)

#     g = nx.Graph()
#     for n in nodes:
#         g.add_node(n, features=node_features[n])
#     g.add_edges_from([(nodes[0], nodes[1]), (nodes[2], nodes[3])])

#     g2 = nx.Graph()
#     for n in nodes:
#         g2.add_node(n, features=node_features[n])
#     g2.add_edges_from([(nodes[1], nodes[2]), (nodes[3], nodes[0])])
#     print(g.edges())
#     print(g2.edges())

#     gfp1 = GraphFingerprint(g, 2, shapes)
#     gfp2 = GraphFingerprint(g2, 2, shapes)

#     fp1 = gfp1.compute_fingerprint(wb.vect, wb.unflattener)
#     fp2 = gfp2.compute_fingerprint(wb.vect, wb.unflattener)

#     assert not np.array_equal(fp1, fp2)


# def test_same_graph_structure_diff_nodes_have_diff_fingerprints():
#     """
#     Given a graph that has the same connectivity structure, e.g. a ring graph
#     of 4 nodes, check that when the actual nodes are different, the final
#     fingerprint is also different.
#     """

#     nodes1 = sample(all_letters, 4)
#     g = nx.Graph()
#     for n in nodes1:
#         g.add_node(n, features=node_features[n])
#     g.add_edges_from([(nodes1[0], nodes1[1]),
#                       (nodes1[1], nodes1[2]),
#                       (nodes1[2], nodes1[3]),
#                       (nodes1[3], nodes1[0])])

#     nodes2 = sample(all_letters, 4)
#     g2 = nx.Graph()
#     for n in nodes2:
#         g2.add_node(n, features=node_features[n])
#     g2.add_edges_from([(nodes2[0], nodes2[1]),
#                       (nodes2[1], nodes2[2]),
#                       (nodes2[2], nodes2[3]),
#                       (nodes2[3], nodes2[0])])

#     gfp1 = GraphFingerprint(g, 2, shapes)
#     gfp2 = GraphFingerprint(g2, 2, shapes)

#     fp1 = gfp1.compute_fingerprint(wb.vect, wb.unflattener)
#     fp2 = gfp2.compute_fingerprint(wb.vect, wb.unflattener)

#     assert not np.array_equal(fp1, fp2)


# def test_shapes_correctly_assigned():
#     """
#     The length of each layer's feature vector can be set independently. Let's
#     make sure that the shapes are assigned correctly between the weights &
#     biases class and the layers.
#     """

#     g = choice(syngraphs)
#     gfp = GraphFingerprint(g, 2, shapes)

#     for layer, graph in gfp.layers.items():
#         if layer != 0:
#             for n, d in graph.nodes(data=True):
#                 feat_shape = d['features'].shape
#                 print(layer, feat_shape)
#                 assert wb[layer]['self_weights'].shape[0] == feat_shape[1]
#                 assert wb[layer-1]['self_weights'].shape[1] == feat_shape[1]
