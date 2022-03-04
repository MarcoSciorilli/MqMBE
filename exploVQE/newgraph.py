from copy import deepcopy
from typing import Optional, Tuple
import networkx as nx
import numpy as np
from qiskit_optimization import QuadraticProgram

from itertools import combinations, groupby


def complete_graph_instantiater(number=50, size=10):
    dummy_graph = nx.complete_graph(size)
    return [weighted_graph(dummy_graph) for i in range(number)]


def quadratic_program_from_graph(graph: nx.Graph) -> QuadraticProgram:
    """Constructs a quadratic program from a given graph for a MaxCut problem instance.
    Args:
        graph: Underlying graph of the problem.
    Returns:
        QuadraticProgram
    """

    # Get weight matrix of graph
    weight_matrix = -nx.adjacency_matrix(graph)
    shape = weight_matrix.shape
    size = shape[0]

    # Build qubo matrix Q from weight matrix W
    qubo_matrix = np.zeros((size, size))
    qubo_vector = np.zeros(size)
    for i in range(size):
        for j in range(size):
            qubo_matrix[i, j] -= weight_matrix[i, j]
    for i in range(size):
        for j in range(size):
            qubo_vector[i] += weight_matrix[i, j]
    my_quadratic_program = QuadraticProgram('my_problem')
    for k in range(size):
        my_quadratic_program.binary_var(name=f'x_{k}')

    quadratic = qubo_matrix
    linear = qubo_vector
    my_quadratic_program.minimize(quadratic=quadratic, linear=linear)

    return my_quadratic_program.to_ising()[0].to_matrix()


def create_graph(index, nodes_number, random, softmax=False):
    p = 1
    if random:
        np.random.seed(index)
        p = np.random.uniform(0, 1)
    dummy_graph = gnp_random_connected_graph(nodes_number, p, index)
    graph = weighted_graph(dummy_graph, seed=index, softmax=softmax)
    return graph


def weighted_graph(graph: nx.Graph, weight_range: Tuple[float, float] = (0, 1), integer_weights: bool = False,
                   seed: Optional[int] = None, softmax=False) -> nx.Graph:
    """
    Takes an unweighted input graph and returns a weighted graph where the weights are uniformly sampled at random
    Args:
        graph: Unweighted graph to add edge weights to
        weight_range: Range of weights to sample from
        integer_weights: Specifies whether weights should be integer (True) or float (False)
        seed: A seed for the random number generator
    Returns:
        The weighted graph
    """
    if seed:
        np.random.seed(seed)

    weighted_graph = deepcopy(graph)
    for edge in weighted_graph.edges:
        if integer_weights:
            weighted_graph[edge[0]][edge[1]]['weight'] = np.random.randint(int(weight_range[0]), int(weight_range[1]))
        else:
            weighted_graph[edge[0]][edge[1]]['weight'] = np.random.uniform(weight_range[0], weight_range[1])
    if softmax:
        B = nx.to_numpy_array(weighted_graph)
        B = softmax_1d(B)
        weighted_graph = nx.from_numpy_array(B)
    return weighted_graph


def softmax_1d(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def softmax_2d(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def gnp_random_connected_graph(n, p, seed):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    np.random.seed(seed)
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = node_edges[np.random.choice(len(node_edges))]
        G.add_edge(*random_edge)
        for e in node_edges:
            if np.random.random() < p:
                G.add_edge(*e)
    return G
