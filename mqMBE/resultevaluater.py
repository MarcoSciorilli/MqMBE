from typing import List
import networkx as nx
import numpy as np
import cvxpy as cvx


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
    for i in range(len(bitstring)):
        if bitstring[i] < 0:
            bitstring[i] = 0
    weight_matrix = nx.adjacency_matrix(graph).toarray()
    size = weight_matrix.shape[0]
    value = 0.
    for i in range(size):
        for j in range(size):
            value += weight_matrix[i, j] * bitstring[i] * (1 - bitstring[j])

    return value

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


def brute_force(w, n):
    cuts ={}
    for b in range(2**n):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
        cost = 0
        for i in range(n):
            for j in range(n):
                cost = cost + w[i,j]*x[i]*(1-x[j])
                cuts[str(x)]=cost
    return [max(cuts.values()), min(cuts.values()), max(cuts, key=cuts.get).replace('0', '-1')]


def brute_force_graph(graph=None):
    adjacency_matrix = nx.adjacency_matrix(graph)
    return brute_force(adjacency_matrix, adjacency_matrix.shape[0])


def goemans_williamson(graph: nx.Graph):
    """
    The Goemans-Williamson algorithm for solving the maxcut problem.
    Ref:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming
    Returns:
        np.ndarray: Graph coloring (+/-1 for each node)
        float:      The GW score for this cut.
        float:      The GW bound from the SDP relaxation
    """

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
    #score = colors @ laplacian @ colors.T
    return [colors.astype(int), maxcut_cost_fn(graph, colors.copy())]