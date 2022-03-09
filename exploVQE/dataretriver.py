import multiprocessing as mp
import numpy as np
from time import time
from exploVQE.newgraph import create_graph, quadratic_program_from_graph
from exploVQE.resultevaluater import classical_solution, get_overlap
from exploVQE.ansatz import var_form, circuit_none, overlap_retriver
from qibo import models, hamiltonians, callbacks, gates
import qibo
import pickle


def classical_soution_finder(starting=0, ending=10, nodes_number=6, random=False):
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
    process_number = 10
    pool = mp.Pool(process_number)
    result = [pool.apply_async(classical_solution, (i, nodes_number, random)) for i in range(starting, ending)]
    pool.close()
    pool.join()
    my_results = [r.get() for r in result]

    with open(
            f'exact_solution_n_{nodes_number}_s_{starting}_e_{ending}_random_{random}.npy',
            'wb') as f:
        pickle.dump(my_results, f)


def overlap_evaluater_parallel(starting=0, ending=10, layer_number=1, nodes_number=6, optimization='COBYLA',
                               graph_list=None, initial_point=False, random=False, entanglement='basic'):
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
    process_number = 30
    pool = mp.Pool(process_number)
    overlaps = np.empty(ending - starting)
    energies = np.empty(ending - starting)
    if graph_list:
        starting = 0
        ending = len(graph_list)
        result_exact = [classical_solution(i, nodes_number, random, graph=graph_list[i]) for i in
                        range(starting, ending)]
        results = [pool.apply_async(single_graph_evaluation, (i, result_exact[i - starting], layer_number,
                                                              optimization, nodes_number, graph_list[i], initial_point,
                                                              random, entanglement)) for i in range(starting, ending)]

    else:
        result_exact = exact_loader(nodes_number, starting, ending, random)
        start = time()
        results = [pool.apply_async(single_graph_evaluation, (i, result_exact[i - starting], layer_number,
                                                              optimization, nodes_number, graph_list, initial_point,
                                                              random, entanglement)) for i in range(starting, ending)]
    average_time = (time() - start) * process_number / (ending - starting)
    my_results = [r.get() for r in results]
    for i in range(len(my_results)):
        overlaps[i] = my_results[i][1]
        energies[i] = my_results[i][2]
    pool.close()
    pool.join()

    file_manager(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, initial_point,
                 random, average_time, entanglement)


def overlap_evaluater(starting=0, ending=10, layer_number=1, nodes_number=6, optimization='COBYLA',
                      graph_list=None,
                      initial_point=False, random=False, entanglement='basic'):
    if nodes_number < 14:
        qibo.set_backend("numpy")
        my_time = time()
        overlap_evaluater_parallel(starting=starting, ending=ending, layer_number=layer_number, nodes_number=nodes_number, optimization=optimization,
                                   graph_list=graph_list, initial_point=initial_point, random=random, entanglement=entanglement)
        print("tempo totale:", time()-my_time)
    else:
        qibo.set_backend("qibotf")
        my_time = time()
        overlap_evaluater_serial(starting=starting, ending=ending, layer_number=layer_number, nodes_number=nodes_number, optimization=optimization,
                                   graph_list=graph_list, initial_point=initial_point, random=random, entanglement=entanglement)
        print("tempo totale:", time() - my_time)


def overlap_evaluater_serial(starting=0, ending=10, layer_number=1, nodes_number=6, optimization='COBYLA',
                      graph_list=None,
                      initial_point=False, random=False, entanglement='basic'):
    """
    Save on file data metrics evaluating VQE overlaps performances averaged over n different weighted graphs
    depending on the number of nodes and layer.
    Args:
        dimension: Number of graphs to average over
        layer_number: Number of layers in VQE
        nodes_number: Number of nodes in the graph
        optimization: Kind of optimization algorithm used by VQE
    Returns:
        None

    """

    overlaps = np.empty(ending - starting)
    energies = np.empty(ending - starting)
    result_exact = exact_loader(nodes_number, starting, ending, random)
    start = time()
    my_results = [
        single_graph_evaluation(i, result_exact[i - starting], layer_number,
                                optimization, nodes_number, graph_list, initial_point,
                                random, entanglement) for i in range(ending - starting)]

    average_time = (time() - start) / (ending - starting)
    for i in range(len(my_results)):
        overlaps[i] = my_results[i][1]
        energies[i] = my_results[i][2]

    file_manager(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, initial_point,
                 random, average_time, entanglement)


def single_graph_evaluation(index, result_exact=None, layer_number=1, optimization='COBYLA', nodes_number=6,
                            graph=None,
                            initial_parameters=None, random=False, entanglement='basic'):
    if graph is None:
        graph = create_graph(index, nodes_number, random)
    quadratic_program = quadratic_program_from_graph(graph)
    right_solution = result_exact[2]
    if initial_parameters:
        initial_parameters = np.random.normal(0.5, 0.01, 3 * layer_number * nodes_number)
    if layer_number > 0:
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
    else:
        circuit = circuit_none(nodes_number)
        outputstate = circuit()
        return get_overlap(right_solution, outputstate.data)


def file_manager(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, initial_point,
                 random, average_time=None, entanglement='basic'):
    with open(
            f'overlap_average_p_{layer_number}_n_{nodes_number}_s_{starting}_e_{ending}_opt_{optimization}_init_{initial_point}_random_{random}_entang_{entanglement}.npy',
            'wb') as f:
        np.save(f, overlaps)
        np.save(f, energies)
        np.save(f, average_time)


def exact_loader(nodes_number, starting, ending, random):
    with open(
            f'exact_solution_n_{nodes_number}_s_{starting}_e_{ending}_random_{random}.npy',
            'rb') as f:
        result = pickle.load(f)
    return result