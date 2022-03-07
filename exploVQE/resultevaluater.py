from typing import List
from typing import Optional, Tuple
import networkx as nx
import numpy as np
from numpy import linalg as la
import cvxpy as cvx
from exploVQE.newgraph import create_graph, quadratic_program_from_graph
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'



def classical_solution(index, nodes_number, random, graph=None):
    if graph is None:
        graph = create_graph(index, nodes_number, random)
    quadratic_program = quadratic_program_from_graph(graph)
    results = la.eig(quadratic_program)
    return [max(results[0]), min(results[0]), results[1][np.where(results[0] == min(results[0]))]]


def classical_solution_no_graph(quadratic_program):
    hamiltonian = quadratic_program.to_ising()[0].to_matrix()
    results = la.eig(hamiltonian)
    return [max(results[0]), min(results[0]), results[1][np.where(results[0] == min(results[0]))]]

def maxcut_cost_fn(graph: nx.Graph, bitstring: List[int]) -> float:
    """
    Computes the maxcut cost function value for a given graph and cut represented by some bitstring
    Args:
        graph: The graph to compute cut values for
        bitstring: A list of integer values '0' or '1' specifying a cut of the graph
    Returns:
        The value of the cut
    """
    # Get the weight matrix of the graph
    weight_matrix = nx.adjacency_matrix(graph).toarray()
    size = weight_matrix.shape[0]
    value = 0.
    for i in range(size):
        for j in range(size):
            value += weight_matrix[i, j] * bitstring[i] * (1 - bitstring[j])

    return value

def maxcut_find_solution(graph: nx.Graph) -> None:
    """
    Return the best cut as a bitstring of the solution and its cost value
    Args:
        graph: The graph to compute cut values for
    """
    num_vars = graph.number_of_nodes()
    # Create list of bitstrings and corresponding cut values
    bitstrings = ['{:b}'.format(i).rjust(num_vars, '0')[::-1] for i in range(2 ** num_vars)]
    values = [maxcut_cost_fn(graph=graph, bitstring=[int(x) for x in bitstring]) for bitstring in bitstrings]
    # Sort both lists by largest cut value
    #values, bitstrings = zip(*sorted(zip(values, bitstrings)))
    return values

def get_overlap(vector_1, vector_2):
    """
    Takes the inner product between two eigenstate (in order). In case of degeneracy, it considers the projection into
    the ground-state subspace.
    Args:
        vector_1: First eigentstate
        vector_2: Second eigenstate
    Returns:
        Real number: inner product between the two
    """
    overlap = 0
    for i in range(len(vector_1)):
        overlap += abs(np.conj(vector_1[i]) @ vector_2)**2
    return overlap


def goemans_williamson(graph: nx.Graph) -> Tuple[np.ndarray, float, float]:
    """
    The Goemans-Williamson algorithm for solving the maxcut problem.
    Ref:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
    Returns:
        np.ndarray: Graph coloring (+/-1 for each node)
        float:      The GW score for this cut.
        float:      The GW bound from the SDP relaxation
    """
    # Kudos: Originally implementation by Nick Rubin, with refactoring and
    # cleanup by Jonathon Ward and Gavin E. Crooks
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # Setup and solve the GW semidefinite programming problem
    psd_mat = cvx.Variable(laplacian.shape, PSD=True)
    obj = cvx.Maximize(cvx.trace(laplacian @ psd_mat))
    constraints = [cvx.diag(psd_mat) == 1]  # unit norm
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT)

    evals, evects = np.linalg.eigh(psd_mat.value)
    sdp_vectors = evects.T[evals > float(1.0E-6)].T

    # Bound from the SDP relaxation
    bound = np.trace(laplacian @ psd_mat.value)

    random_vector = np.random.randn(sdp_vectors.shape[1])
    random_vector /= np.linalg.norm(random_vector)
    colors = np.sign([vec @ random_vector for vec in sdp_vectors])
    score = colors @ laplacian @ colors.T

    return colors, score, bound