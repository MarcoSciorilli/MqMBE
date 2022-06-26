from copy import deepcopy
from typing import Optional, Tuple, List
import networkx as nx
import numpy as np
from itertools import combinations, groupby


class RandomGraphs(object):
    def __init__(self, index: int, nodes_number: int, true_random: Optional[bool] = False,
                 softmax: Optional[bool] = False, fully_connected = False):
        self.index = index
        self.nodes_number = nodes_number
        self.fully_connected = fully_connected
        if true_random:
            np.random.seed(np.randint(0))
            self.index = np.randint(0)
        self.softmax = softmax
        self.graph = self.create_graph()

    @staticmethod
    def complete_graph_instantiater(number: int = 50, size: int = 10) -> List:
        """
        Function which create a list of fully connected graphs with random weights.
        :param number: Number of graphs in the list
        :param size: Number of nodes in each graph
        :return: List of nx graphs
        """
        dummy_graph = nx.complete_graph(size)
        return [RandomGraphs.weighted_graph(graph=dummy_graph) in range(number)]

    @staticmethod
    def weighted_graph(graph: nx.Graph, weight_range: Optional[Tuple[float, float]] = (-10.0, 10.0),
                       integer_weights: Optional[bool] = True,
                       seed: Optional[int] = None, softmax=False) -> nx.Graph:
        """
        Takes an unweighted input graph and returns a weighted graph where the weights are uniformly sampled at random
        :param graph: The graph to weight
        :param weight_range: The range of values for the weights
        :param integer_weights: True if only integer weight are considered
        :param seed: Seed for the generation of the random weight
        :param softmax: True to map the weigths into a softmax
        :return: The weighted graph
        """
        # If seed is given, fix the seed
        if seed:
            np.random.seed(seed)
        # Do a deepcopy of the graph
        weighted_graph = deepcopy(graph)
        # Per each edge, assign a weight
        for edge in weighted_graph.edges:
            if integer_weights:
                weighted_graph[edge[0]][edge[1]]['weight'] = np.random.randint(int(weight_range[0]),int(weight_range[1]))
            else:
                weighted_graph[edge[0]][edge[1]]['weight'] = np.random.uniform(weight_range[0], weight_range[1])
        # If softmax is True, perform a softmax transformation of the weights
        if softmax:
            B = nx.to_numpy_array(weighted_graph)
            B = RandomGraphs.softmax_1d(B)
            weighted_graph = nx.from_numpy_array(B)
        return weighted_graph

    @staticmethod
    def softmax_1d(array: np.array) -> np.array:
        """
        Perform a softmax transformation of a numpy array
        :param array: the numpy array
        :return: the softmax numpy array
        """
        y = np.exp(array - np.max(array))
        f_x = y / np.sum(np.exp(array))
        return f_x

    @staticmethod
    def gnp_random_connected_graph(nodes_number: int, prob: float, seed: int) -> nx.Graph:
        """
        Generates a random undirected graph, similarly to an Erdős-Rényi
        graph, but enforcing that the resulting graph is conneted.
        :param nodes_number:Number of nodes
        :param prob: Probability of each edge to exist
        :param seed:
        :return: A connected graph
        """
        # Fix the passed seed
        np.random.seed(seed)
        edges = combinations(range(nodes_number), 2)
        G = nx.Graph()
        G.add_nodes_from(range(nodes_number))
        if prob <= 0:
            return G
        if prob >= 1:
            return nx.complete_graph(nodes_number, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = node_edges[np.random.choice(len(node_edges))]
            G.add_edge(*random_edge)
            for e in node_edges:
                if np.random.random() < prob:
                    G.add_edge(*e)
        return G

    def create_graph(self) -> nx.Graph:
        """
        Function which create a random connected graph.
        :return: A graph
        """
        # Initialise the probability of each node of having an edge with a neighbour
        if self.fully_connected:
            p = 1
        else:
            # Fix the probability if the same seed is used
            np.random.seed(self.index)
            p = np.random.uniform(0, 1)
        # Create the graph unweighted
        dummy_graph = self.gnp_random_connected_graph(self.nodes_number, p, self.index)
        # Assign weight to the edges
        graph = self.weighted_graph(dummy_graph, seed=self.index, softmax=self.softmax)
        return graph

    def return_index(self):
        return self.index