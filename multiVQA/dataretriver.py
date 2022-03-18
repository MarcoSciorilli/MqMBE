import multiprocessing as mp
import numpy as np
from time import time
from multiVQA.newgraph import RandomGraphs
from multiVQA.resultevaluater import brute_force_graph
from multiVQA.ansatz import var_form, var_form_RY
from multiVQA.multibase import MultibaseVQA
from qibo import models, hamiltonians, callbacks, gates
import qibo
import pickle
import networkx as nx
from multiVQA.datamanager import insert_value_table, connect_database, create_table, read_data


def single_graph_evaluation(kind, instance,  nodes_number,layer_number, optimization, initial_parameters, compression, pauli_string_length, entanglement, graph_list, graph_kind):
    if graph_list is None:
        true_random_graphs = False
        if graph_kind == 'random':
            true_random_graphs = True
        graph = RandomGraphs(instance, nodes_number, true_random_graphs).graph
        if true_random_graphs:
            instance = graph.return_index()
    else:
        graph = graph_list[instance]
    if kind == 'bruteforce':
        my_time = time()
        result = brute_force_graph(graph)
        timing = my_time - time()
        max_energy, min_energy, solution, energy_ratio = result[0], result[1], result[2], 1
        qubits, compression,  pauli_string_length, epochs, parameters = 'None', 'None', 'None', 'None', 'None'
    else:
        adjacency_matrix = graph_to_dict(graph)
        result_exact = get_exact_solution('MaxCutDatabase', 'MaxCutDatabase', ['max_energy','min_energy'], {'kind': 'bruteforce', 'instance': instance, 'nodes_number': nodes_number, 'graph_kind': graph_kind})
        if kind == 'multibaseVQA':
            qubits = MultibaseVQA.get_num_qubits(nodes_number, pauli_string_length, compression)
            circuit = var_form(qubits, layer_number, entanglement)
            if initial_parameters == 'None':
                initial_parameters = np.random.normal(0, 1, len(circuit.get_parameters(format='flatlist')))
            solver = MultibaseVQA(circuit, adjacency_matrix)
            solver.encode_nodes(nodes_number, pauli_string_length, compression)
            solver.set_activation(np.tanh)
            my_time = time()
            result, cut, parameters, extra = solver.minimize(initial_parameters, method=optimization, tol=1.11e-6)
            timing = my_time - time()
            max_energy, min_energy, epochs = cut, 'None', extra['nfev']
            energy_ratio = (cut - result_exact[0][1]) / (result_exact[0][0] - result_exact[0][1])
            # Da aggiungere
            solution = 'None'

        if kind == 'classicVQE':
            qubits = nodes_number

    row = {'kind':  kind, 'instance': instance, 'solution': solution, 'max_energy': max_energy, 'min_energy': min_energy, 'energy_ratio': energy_ratio,  'layer_number': layer_number, 'nodes_number': nodes_number, 'optimization': optimization, 'initial_parameters': str(initial_parameters.tolist()), 'qubits': qubits, 'compression': compression, 'pauli_string_length': pauli_string_length, 'entanglement': entanglement, 'epochs': epochs, 'time': timing, 'parameters': str(parameters.tolist()), 'graph_kind': graph_kind}
    insert_value_table('MaxCutDatabase', 'MaxCutDatabase', row)




def benchmarker(kind,  nodes_number, starting, ending, layer_number='None', optimization= 'None', initial_parameters = 'None', compression= 'None', pauli_string_length = 'None', entanglement = 'None', graph_list=None, graph_kind='indexed'):
    if nodes_number < 100:
        qibo.set_backend("numpy")
        my_time = time()
        eignensolver_evaluater_parallel(kind, nodes_number, starting, ending, layer_number, optimization, initial_parameters, compression, pauli_string_length, entanglement, graph_list, graph_kind)
        print("tempo totale:", time()-my_time)
    else:
        qibo.set_backend("qibotf")
        my_time = time()
        #VQE_evaluater_serial(starting=starting, ending=ending, layer_number=layer_number, nodes_number=nodes_number, optimization=optimization,
        #                           graph_list=graph_list, pick_init_parameter=pick_init_parameter, random_graphs=random_graphs, entanglement=entanglement,  multibase=multibase)
        print("tempo totale:", time() - my_time)


def eignensolver_evaluater_parallel(kind, nodes_number, starting, ending, layer_number, optimization, initial_parameters, compression, pauli_string_length, entanglement, graph_list, graph_kind):
    process_number = 30
    pool = mp.Pool(process_number)
    if graph_list != None:
        # Da sistemare
        starting = 0
        ending = len(graph_list)
        # Mettere soluzione classica ricorsiva
        result_exact = [classical_solution(i, nodes_number, random_graphs, graph=graph_list[i]) for i in
                        range(starting, ending)]
    results = [pool.apply_async(single_graph_evaluation, (kind, instance,  nodes_number,layer_number, optimization, initial_parameters, compression, pauli_string_length, entanglement, graph_list, graph_kind)) for instance in range(starting, ending)]
    pool.close()
    pool.join()

def initialize_database(name_database):
    rows = {'kind': 'TEXT', 'instance': 'TEXT', 'solution': 'TEXT', 'max_energy': 'FLOAT', 'min_energy': 'FLOAT', 'energy_ratio': 'FLOAT', 'layer_number': 'INT', 'nodes_number': 'INT', 'optimization': 'TEXT', 'initial_parameters': 'TEXT', 'qubits': 'INT', 'compression': 'FLOAT', 'pauli_string_length': 'INT', 'entanglement': 'TEXT', 'epochs': 'INT', 'time': 'FLOAT', 'parameters': 'TEXT', 'graph_kind': 'TEXT'}
    connect_database(name_database)
    create_table(name_database, name_database, rows)

def get_exact_solution(name_database, name_table, data_to_read, parameters_to_fix):
    return read_data(name_database, name_table, data_to_read, parameters_to_fix)










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



def single_graph_evaluation_classic(index, result_exact=None, layer_number=1, optimization='COBYLA', nodes_number=6,
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

def graph_to_dict(graph):
    adjacency_matrix=nx.to_numpy_array(graph)
    edges = {}
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i):
            if adjacency_matrix[i][j] ==0:
                continue
            edges[(i,j)]=adjacency_matrix[i][j]
    return edges


def exact_loader(nodes_number, starting, ending, random):
    with open(
            f'exact_solution_n_{nodes_number}_s_{starting}_e_{ending}_random_{random}_hamiltonian_False.npy',
            'rb') as f:
        result = pickle.load(f)
    return result