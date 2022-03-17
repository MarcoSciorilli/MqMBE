import multiprocessing as mp
import numpy as np
from time import time
from multiVQA.newgraph import RandomGraphs
from multiVQA.resultevaluater import classical_solution, brute_force_random_graph
from multiVQA.ansatz import var_form, var_form_RY
from multiVQA.multibase import MultibaseVQA
from qibo import models, hamiltonians, callbacks, gates
import qibo
import pickle
import networkx as nx
from multiVQA.datamanager import insert_value_table, connect_database, create_table


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
            f'exact_solution_n_{nodes_number}_s_{starting}_e_{ending}_random_{random}.npy',
            'wb') as f:
        pickle.dump(my_results, f)


def VQE_evaluater(starting=0, ending=10, layer_number=1, nodes_number=6, optimization='COBYLA',
                      graph_list=None,
                      pick_init_parameter=True, random_graphs=False, entanglement='basic', multibase=False):
    if nodes_number < 100:
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

    process_number = 30
    pool = mp.Pool(process_number)
    overlaps = np.empty(ending - starting)
    energies = np.empty(ending - starting)
    compression = 1
    pauli_string_length = 1
    if multibase:
        qubits = MultibaseVQA.get_num_qubits(nodes_number, pauli_string_length, compression)
    if graph_list:
        starting = 0
        ending = len(graph_list)
        result_exact = [classical_solution(i, nodes_number, random_graphs, graph=graph_list[i]) for i in
                        range(starting, ending)]
        if multibase:
            results = [pool.apply_async(single_graph_evaluation_multibase, (i, qubits, pauli_string_length, compression, result_exact[i - starting], layer_number,
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
            results = [pool.apply_async(single_graph_evaluation_multibase, (i,qubits, pauli_string_length, compression, result_exact[i - starting], layer_number,
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
    if multibase:
        file_manager_multibase(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, pick_init_parameter,
                 random_graphs, qubits, compression, pauli_string_length, entanglement)
    else:
        file_manager(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, pick_init_parameter,
                 random_graphs, average_time, entanglement)



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
                 random_graphs, average_time, entanglement)


def single_graph_evaluation(index, result_exact=None, layer_number=1, optimization='COBYLA', nodes_number=6,
                            graph=None,
                            pick_init_parameter=False, random_graphs=False, entanglement='basic'):
    if graph is None:
        graph = RandomGraphs(index, nodes_number, random_graphs).create_graph()
    quadratic_program = RandomGraphs.quadratic_program_from_graph(graph)
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

def single_graph_evaluation(kind, instance, layer_number, nodes_number, optimization= 'COBYLA', initial_parameters = None, compression= 'None', pauli_string_length = 'None', entanglement = 'basic', graph=None, True_random_graphs=False):
    if graph is None:
            graph = RandomGraphs(instance, nodes_number, True_random_graphs).graph
            if True_random_graphs:
                instance = graph.return_index()
    else:
        # Add choosen graph
        pass
    if kind == 'bruteforce':

    else:
        if kind == 'multibaseVQA':
        if kind == 'classicVQE':



    row = {'kind':  kind, 'instance': instance, 'solution': solution, 'max_energy': max_energy, 'min_energy': min_energy, 'energy_ratio': energy_ratio , 'overlap': overlap, 'layer_number': layer_number, 'nodes_number': nodes_number, 'optimization': optimization, 'initial_parameters': initial_parameters, 'qubits': qubits, 'compression': compression, 'pauli_string_length': pauli_string_length, 'entanglement': entanglement, 'epochs': epochs, 'time': time, 'parameters': parameters}
    insert_value_table('MaxCutData', 'MaxCutData', row)

def single_graph_evaluation_multibase(index, qubits, pauli_string_length, compression, result_exact=None, layer_number=1, optimization='COBYLA', nodes_number=6,
                            graph=None,
                            initial_parameters=None, True_random_graphs=False, entanglement='basic'):

    if graph is None:
            graph = RandomGraphs(index, nodes_number, True_random_graphs).graph
            if True_random_graphs:
                index = graph.return_index()

    circuit = var_form(qubits, layer_number, entanglement)
    if initial_parameters is None:
        initial_parameters = np.random.normal(0, 1, len(circuit.get_parameters(format='flatlist')))
    adjacency_matrix = graph_to_dict(graph)
    solver = MultibaseVQA(circuit, adjacency_matrix)
    solver.encode_nodes(nodes_number, pauli_string_length, compression)
    solver.set_activation(np.tanh)
    result, cut, parameters, extra = solver.minimize(initial_parameters, method=optimization, tol=1.11e-6)
    print(result)
    max_energy = cut
    min_energy = 'None'
    energy_ratio = (cut - result_exact[1]) / (result_exact[0] - result_exact[1])
    overlap = 'None'
    kind = 'Multibase'
    instance = f'{index}'
    solution = 'None'
    epochs = 'None'
    time = 'None'
    initial_parameters = 'None'
    parameters = 'None'
    row = {'kind':  kind, 'instance': instance, 'solution': solution, 'max_energy': max_energy, 'min_energy': min_energy, 'energy_ratio': energy_ratio , 'overlap': overlap, 'layer_number': layer_number, 'nodes_number': nodes_number, 'optimization': optimization, 'initial_parameters': initial_parameters, 'qubits': qubits, 'compression': compression, 'pauli_string_length': pauli_string_length, 'entanglement': entanglement, 'epochs': epochs, 'time': time, 'parameters': parameters}
    insert_value_table('MaxCutData', 'MaxCutData', row)

    return [index, 0, (cut - result_exact[1]) / (result_exact[0] - result_exact[1])]

def initialize_database(name_database):
    rows = {'kind': 'TEXT', 'instance': 'TEXT', 'solution': 'TEXT', 'max_energy': 'FLOAT', 'min_energy': 'FLOAT', 'energy_ratio': 'FLOAT' , 'overlap': 'FLOAT', 'layer_number': 'INT', 'nodes_number': 'INT', 'optimization': 'TEXT', 'initial_parameters': 'TEXT', 'qubits': 'INT', 'compression': 'FLOAT', 'pauli_string_length': 'INT', 'entanglement': 'TEXT', 'epochs': 'INT', 'time': 'FLOAT', 'parameters': 'TEXT'}
    connect_database(name_database)
    create_table(name_database, name_database, rows)

def graph_to_dict(graph):
    adjacency_matrix=nx.to_numpy_array(graph)
    edges = {}
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i):
            if adjacency_matrix[i][j] ==0:
                continue
            edges[(i,j)]=adjacency_matrix[i][j]
    return edges

def file_manager(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, initial_point,
                 random, average_time=None, entanglement='basic'):
    with open(
        f'overlap_average_p_{layer_number}_n_{nodes_number}_s_{starting}_e_{ending}_opt_{optimization}_init_{initial_point}_random_{random}_entang_{entanglement}.npy',
            'wb') as f:
        np.save(f, overlaps)
        np.save(f, energies)
        np.save(f, average_time)

def file_manager_multibase(overlaps, energies, layer_number, nodes_number, starting, ending, optimization, initial_point,
                 random, qubits, compression, pauli_string_length, entanglement):
    with open(
        f'cut_average_p_{layer_number}_n_{nodes_number}_s_{starting}_e_{ending}_opt_{optimization}_init_{initial_point}_random_{random}_entang_{entanglement}_qb_{qubits}_comp_{compression}_pauli_{pauli_string_length}.npy',
            'wb') as f:
        np.save(f, overlaps)
        np.save(f, energies)



def exact_loader(nodes_number, starting, ending, random):
    with open(
            f'exact_solution_n_{nodes_number}_s_{starting}_e_{ending}_random_{random}_hamiltonian_False.npy',
            'rb') as f:
        result = pickle.load(f)
    return result