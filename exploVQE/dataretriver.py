import multiprocessing as mp
import numpy as np
from time import time
from exploVQE.newgraph import create_graph, quadratic_program_from_graph
from exploVQE.resultevaluater import classical_solution, brute_force_random_graph
from exploVQE.ansatz import var_form, var_form_RY
from exploVQE.multibase import MultibaseVQA
from qibo import models, hamiltonians, callbacks, gates
import qibo
import pickle
import networkx as nx

def classical_solution_finder(starting=0, ending=10, nodes_number=6, random=True, hamiltonian=False):
    """
    Save on file data metrics evaluating QAOA overlaps performances averaged over n different weighted graphs
    depending on the number of nodes and layer.
    Args:
        nodes_number: Number of nodes in the graph
    Returns:
        None

    """
    process_number = 10
    pool = mp.Pool(process_number)
    if hamiltonian:
        result = [pool.apply_async(classical_solution, (i, nodes_number, random)) for i in range(starting, ending)]
    else:
        result = [pool.apply_async(brute_force_random_graph, (i, nodes_number, random)) for i in range(starting, ending)]
    pool.close()
    pool.join()
    my_results = [r.get() for r in result]

    with open(
            f'exact_solution_n_{nodes_number}_s_{starting}_e_{ending}_random_{random}_hamiltonian_{hamiltonian}.npy',
            'wb') as f:
        pickle.dump(my_results, f)


def VQE_evaluater(starting=0, ending=10, layer_number=1, nodes_number=6, optimization='COBYLA',
                      graph_list=None,
                      pick_init_parameter=True, random_graphs=False, entanglement='basic', multibase=False):
    if nodes_number < 14:
        print("ARRIVO QUI")
        qibo.set_backend("numpy")
        my_time = time()
        VQE_evaluater_parallel(starting=starting, ending=ending, layer_number=layer_number, nodes_number=nodes_number, optimization=optimization,
                                   graph_list=graph_list, pick_init_parameter=pick_init_parameter, random_graphs=random_graphs, entanglement=entanglement, multibase=multibase)
        print("tempo totale:", time()-my_time)
    else:
        qibo.set_backend("qibotf")
        my_time = time()
        VQE_evaluater_serial(starting=starting, ending=ending, layer_number=layer_number, nodes_number=nodes_number, optimization=optimization,
                                   graph_list=graph_list, pick_init_parameter=pick_init_parameter, random_graphs=random_graphs, entanglement=entanglement,  multibase=multibase)
        print("tempo totale:", time() - my_time)


def VQE_evaluater_parallel(starting=0, ending=10, layer_number=1, nodes_number=6, optimization='COBYLA',
                               graph_list=None, pick_init_parameter=True, random_graphs=False, entanglement='basic', multibase=False):
    """
    Save on file data metrics evaluating QAOA overlaps performances averaged over n different weighted graphs
    depending on the number of nodes and layer.
    Args:
        layer_number: Number of layers in QAOA
        nodes_number: Number of nodes in the graph
        optimization: Kind of optimization algorithm used by QAOA
    Returns:
        None

    """
    process_number = 1
    pool = mp.Pool(process_number)
    overlaps = np.empty(ending - starting)
    energies = np.empty(ending - starting)
    if graph_list:
        starting = 0
        ending = len(graph_list)
        result_exact = [classical_solution(i, nodes_number, random_graphs, graph=graph_list[i]) for i in
                        range(starting, ending)]
        if multibase:
            results = [pool.apply_async(single_graph_evaluation_multibase, (i, result_exact[i - starting], layer_number,
                                                                  optimization, nodes_number, graph_list[i], pick_init_parameter,
                                                                  random_graphs, entanglement)) for i in range(starting, ending)]
        else:
            results = [pool.apply_async(single_graph_evaluation, (i, result_exact[i - starting], layer_number,
                                                                  optimization, nodes_number, graph_list[i], pick_init_parameter,
                                                                  random_graphs, entanglement)) for i in range(starting, ending)]

    else:
        result_exact = exact_loader(nodes_number, starting, ending, random_graphs)
        start = time()
        if multibase:
            results = [pool.apply_async(single_graph_evaluation_multibase, (i, result_exact[i - starting], layer_number,
                                                                  optimization, nodes_number, graph_list, pick_init_parameter,
                                                                  random_graphs, entanglement)) for i in range(starting, ending)]
        else:
            results = [pool.apply_async(single_graph_evaluation, (i, result_exact[i - starting], layer_number,
                                                                  optimization, nodes_number, graph_list, pick_init_parameter,
                                                                  random_graphs, entanglement)) for i in range(starting, ending)]

    average_time = (time() - start) * process_number / (ending - starting)
    my_results = [r.get() for r in results]
    for i in range(len(my_results)):
        overlaps[i] = my_results[i][1]
        energies[i] = my_results[i][2]
    pool.close()
    pool.join()

    file_manager(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, pick_init_parameter,
                 random_graphs, average_time, entanglement, multibase)



def VQE_evaluater_serial(starting=0, ending=10, layer_number=1, nodes_number=6, optimization='COBYLA',
                      graph_list=None,
                      pick_init_parameter=True, random_graphs=False, entanglement='basic', multibase=False):
    """
    Save on file data metrics evaluating VQE overlaps performances averaged over n different weighted graphs
    depending on the number of nodes and layer.
    Args:
        layer_number: Number of layers in VQE
        nodes_number: Number of nodes in the graph
        optimization: Kind of optimization algorithm used by VQE
    Returns:
        None

    """

    overlaps = np.empty(ending - starting)
    energies = np.empty(ending - starting)
    result_exact = exact_loader(nodes_number, starting, ending, random_graphs)
    start = time()
    if multibase:
        my_results = [
            single_graph_evaluation_multibase(i, result_exact[i - starting], layer_number,
                                    optimization, nodes_number, graph_list, pick_init_parameter,
                                    random_graphs, entanglement) for i in range(ending - starting)]
    else:
        my_results = [
            single_graph_evaluation(i, result_exact[i - starting], layer_number,
                                    optimization, nodes_number, graph_list, pick_init_parameter,
                                    random_graphs, entanglement) for i in range(ending - starting)]

    average_time = (time() - start) / (ending - starting)
    for i in range(len(my_results)):
        overlaps[i] = my_results[i][1]
        energies[i] = my_results[i][2]

    file_manager(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, pick_init_parameter,
                 random_graphs, average_time, entanglement, multibase)


def single_graph_evaluation(index, result_exact=None, layer_number=1, optimization='COBYLA', nodes_number=6,
                            graph=None,
                            pick_init_parameter=False, random_graphs=False, entanglement='basic'):
    if graph is None:
        graph = create_graph(index, nodes_number, random_graphs)
    quadratic_program = quadratic_program_from_graph(graph)
    right_solution = result_exact[2]
    if pick_init_parameter:
        initial_parameters = np.random.normal(0.5, 0.01, 3 * layer_number * nodes_number)
    else:
        initial_parameters = None
    circuit = var_form(nodes_number, layer_number, entanglement)
    hamiltonian = hamiltonians.Hamiltonian(nodes_number, quadratic_program)
    solver = models.VQE(circuit, hamiltonian)
    result, params, extra = solver.minimize(initial_parameters, method=optimization, compile=False, tol=1.11e-6)
    circuit = var_form(nodes_number, layer_number)
    overlap = callbacks.Overlap(right_solution)
    circuit.add(gates.CallbackGate(overlap))
    circuit.set_parameters(params)
    circuit()
    return [index, float(overlap[0]), (result - result_exact[0]) / (result_exact[1] - result_exact[0])]

def single_graph_evaluation_multibase(index, result_exact=None, layer_number=1, optimization='COBYLA', nodes_number=6,
                            graph=None,
                            pick_init_parameter=False, random_graphs=False, entanglement='basic'):
    if graph is None:
        graph = create_graph(index, nodes_number, random_graphs)
    adjacency_matrix = nx.adjacency_matrix(graph)
    nodes_number = int(nodes_number / 2)
    if pick_init_parameter:
        ### da sistemare
        initial_parameters = np.random.normal(0, 1,  (layer_number+1) * nodes_number)
    else:
        initial_parameters = None
    circuit = var_form_RY(nodes_number, layer_number, entanglement)
    solver = MultibaseVQA(circuit, adjacency_matrix)
    solver.encode_nodes(nodes_number, 1, 1 / 3)
    solver.set_activation(np.tanh)
    result, cut, parameters, extra = solver.minimize(initial_parameters, method=optimization, tol=1.11e-6)
    print(cut, float(result_exact[1]))
    print((cut - result_exact[0]) / (result_exact[1] - result_exact[0]))
    return [index, 0, (cut - result_exact[0]) / (result_exact[1] - result_exact[0])]


def file_manager(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, initial_point,
                 random, average_time=None, entanglement='basic', multibase=False):
    with open(
        f'overlap_average_p_{layer_number}_n_{nodes_number}_s_{starting}_e_{ending}_opt_{optimization}_init_{initial_point}_random_{random}_entang_{entanglement}_multibase_{multibase}.npy',
            'wb') as f:
        np.save(f, overlaps)
        np.save(f, energies)
        np.save(f, average_time)


def exact_loader(nodes_number, starting, ending, random):
    with open(
            f'exact_solution_n_{nodes_number}_s_{starting}_e_{ending}_random_{random}_hamiltonian_False.npy',
            'rb') as f:
        result = pickle.load(f)
    return result