import pickle as pkl
from itertools import combinations

import networkx as nx
import numpy as np


def load_train_data(data_path: str, labels_path: str):
    """Creates two lists with train graphs and their corresponding labels.
    Args:
        data_path (str): Path to the pickle train file.
        labels_path (str): Path to the labels file

    Returns:
        tuple: Lists of nx.Graph elements and ints.
    """
    with open(data_path, "rb") as file:
        train_graphs = pkl.load(file)
    with open(labels_path, "rb") as file:
        train_labels = pkl.load(file)

    # Removes graphs with no edges
    N = len(train_graphs)
    no_edges = (np.array([g.number_of_edges() for g in train_graphs]) == 0).nonzero()[0]
    train_graphs = [train_graphs[el] for el in range(N) if el not in no_edges]
    train_labels = [train_labels[el] for el in range(N) if el not in no_edges]

    return train_graphs, train_labels


def list_nodes_and_labels(g):
    """
    Returns two lists of nodes and labels in graph g.
    """
    nodes = np.array(list(g.nodes()))
    labels = np.array([g[1]["labels"][0] for g in g.nodes(data=True)])
    return nodes, labels


def product_graph(g1, g2):
    """
    Computes the graph product between g1 and g2.
    """
    g1_nodes, g1_labels = list_nodes_and_labels(g1)
    g2_nodes, g2_labels = list_nodes_and_labels(g2)
    product_labels = set(g1_labels)
    product_labels.update(g2_labels)
    product = nx.Graph()
    for label in product_labels:
        g1_idx = (g1_labels == label).nonzero()[0]
        g2_idx = (g2_labels == label).nonzero()[0]

        # Does not add any node to the product graph if label is unique to one of them
        if not len(g2_idx) or not len(g2_idx):
            continue

        # Adds the nodes to the product graph
        for v1 in g1_nodes[g1_idx]:
            for v2 in g2_nodes[g2_idx]:
                product.add_node((v1, v2))

    # Connects the edges
    pairs = list(combinations(product.nodes, 2))
    for p in pairs:
        u, v = p  # Two nodes in the product graph
        u1, u2 = u  # in G1xG2
        v1, v2 = v  # in G1xG2
        if g1.has_edge(u1, v1) and g2.has_edge(u2, v2):
            product.add_edge(u, v)
    return product
