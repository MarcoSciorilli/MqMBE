import multiprocessing as mp
import numpy as np
from time import time
from multiVQA.newgraph import RandomGraphs
from multiVQA.resultevaluater import brute_force_graph, goemans_williamson
from multiVQA.ansatz import var_form
from multiVQA.multibase import MultibaseVQA
import qibo
import networkx as nx
from multiVQA.datamanager import insert_value_table, connect_database, create_table, read_data

def single_graph_evaluation(kind, instance, trial, nodes_number, layer_number, optimization, initial_parameters,
                            compression, pauli_string_length, entanglement, graph, graph_kind, activation_function):
    if graph is None:
        true_random_graphs = False

        if graph_kind == 'random':
            true_random_graphs = True
        graph = RandomGraphs(instance, nodes_number, true_random_graphs).graph
        if true_random_graphs:
            instance = graph.return_index()

    if kind == 'bruteforce':
        my_time = time()
        result = brute_force_graph(graph)
        timing = my_time - time()
        max_energy, min_energy, solution, energy_ratio = result[0], result[1], result[2], 1
        qubits, compression, pauli_string_length, epochs, parameters, number_parameters, activation_function, unrounded_solution = 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None'
    else:
        np.random.seed(trial)
        result_exact = get_exact_solution('MaxCutDatabase', 'MaxCutDatabase', ['max_energy', 'min_energy'],
                                          {'kind': 'bruteforce', 'instance': instance, 'nodes_number': nodes_number,
                                           'graph_kind': graph_kind})

        if len(result_exact) == 0:
            result_exact = [(1, 0)]
        if kind == 'goemans_williamson':
            my_time = time()
            result = goemans_williamson(graph)
            timing = my_time - time()
            max_energy, solution = result[1], str(result[0])
            energy_ratio = (max_energy - result_exact[0][1]) / (result_exact[0][0] - result_exact[0][1])
            qubits, compression, pauli_string_length, epochs, parameters, number_parameters, activation_function, unrounded_solution, min_energy = 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None'
        else:
            adjacency_matrix = graph_to_dict(graph)
            if kind == 'classicVQE':
                pauli_string_length = 1
                compression = 1 / 3
                activation_function = MultibaseVQA.linear_activation()

            qubits = MultibaseVQA.get_num_qubits(nodes_number, pauli_string_length, compression)
            circuit = var_form(qubits, layer_number, entanglement)

            if initial_parameters == 'None':
                initial_parameters = np.random.normal(0, 1, len(circuit.get_parameters(format='flatlist')))
            solver = MultibaseVQA(circuit, adjacency_matrix)
            solver.encode_nodes(nodes_number, pauli_string_length, compression)
            solver.set_activation(activation_function)
            my_time = time()
            result, cut, parameters, extra, unrounded_solution, solution = solver.minimize(initial_parameters, method=optimization, options={'maxiter':1000000000})
            timing = time() - my_time
            print(timing)
            if optimization == 'cma':
                epochs = extra[1].result[3]
            else:
                epochs = extra['nfev']

            max_energy, min_energy, number_parameters, initial_parameters, parameters, unrounded_solution, solution = cut, 'None', len(
                initial_parameters), str(initial_parameters.tolist()), str(parameters.tolist()), str(
                unrounded_solution), str(solution)
            energy_ratio = (cut - result_exact[0][1]) / (result_exact[0][0] - result_exact[0][1])
            activation_function = activation_function.__name__
        if max_energy == energy_ratio:
            energy_ratio = 'None'

    row = {'kind': kind, 'instance': instance, 'trial': trial, 'layer_number': layer_number,
           'nodes_number': nodes_number, 'optimization': optimization, 'activation_function': str(activation_function),
           'compression': compression, 'pauli_string_length': pauli_string_length, 'entanglement': entanglement,
           'graph_kind': graph_kind, 'qubits': qubits, 'solution': solution, 'unrounded_solution': unrounded_solution,
           'max_energy': max_energy, 'min_energy': min_energy, 'energy_ratio': energy_ratio,
           'initial_parameters': initial_parameters, 'parameters': parameters, 'number_parameters': number_parameters,
           'epochs': epochs, 'time': timing}
    insert_value_table('MaxCutDatabase', 'MaxCutDatabase', row)


def benchmarker(kind, nodes_number, starting, ending, trials=1, layer_number='None', optimization='None',
                initial_parameters='None', compression='None', pauli_string_length='None', entanglement='None',
                graph_dict=None, graph_kind='indexed', activation_function=np.tanh):
    if nodes_number < 100:
        qibo.set_backend("numpy")
        my_time = time()
        eignensolver_evaluater_parallel(kind, nodes_number, starting, ending, trials, layer_number, optimization,
                                        initial_parameters, compression, pauli_string_length, entanglement, graph_dict,
                                        graph_kind, activation_function)
        print("Total time:", time() - my_time)
    else:
        qibo.set_backend("numpy")
        my_time = time()
        eignensolver_evaluater_serial(kind, nodes_number, starting, ending, trials, layer_number, optimization,
                                      initial_parameters, compression, pauli_string_length, entanglement, graph_dict,
                                      graph_kind, activation_function)
        print("Total time:", time() - my_time)


def eignensolver_evaluater_parallel(kind, nodes_number, starting, ending, trials, layer_number, optimization,
                                    initial_parameters, compression, pauli_string_length, entanglement, graph_dict,
                                    graph_kind, activation_function):
    process_number = 35
    pool = mp.Pool(process_number)
    if graph_dict is not None:
        results = [pool.apply_async(single_graph_evaluation, (
            'bruteforce', instance, trials, nodes_number, layer_number, optimization, initial_parameters, compression,
            pauli_string_length, entanglement, graph_dict[instance], instance, activation_function)) for instance in
                   graph_dict if graph_dict[instance].number_of_nodes() < 20]
    for trial in range(trials):
        results = [pool.apply_async(single_graph_evaluation, (
            kind, instance, trial, nodes_number, layer_number, optimization, initial_parameters, compression,
            pauli_string_length, entanglement, graph_dict, graph_kind, activation_function)) for instance in
                   range(starting, ending)]
    pool.close()
    pool.join()


def eignensolver_evaluater_serial(kind, nodes_number, starting, ending, trials, layer_number, optimization,
                                  initial_parameters, compression, pauli_string_length, entanglement, graph_dict,
                                  graph_kind, activation_function):
    if graph_dict is not None:
        for instance in graph_dict:
            if graph_dict[instance].number_of_nodes() < 20:
                single_graph_evaluation(
                    'bruteforce', instance, trials, nodes_number, layer_number, optimization, initial_parameters,
                    compression,
                    pauli_string_length, entanglement, graph_dict[instance], instance, activation_function)
    for trial in range(trials):
        for instance in range(starting, ending):
            single_graph_evaluation(kind, instance, trial, nodes_number, layer_number, optimization, initial_parameters,
                                    compression, pauli_string_length, entanglement, graph_dict, graph_kind,
                                    activation_function)


def initialize_database(name_database):
    rows = {'kind': 'TEXT', 'instance': 'TEXT', 'trial': 'INT', 'layer_number': 'INT', 'nodes_number': 'INT',
            'optimization': 'TEXT', 'compression': 'FLOAT', 'pauli_string_length': 'INT', 'entanglement': 'TEXT',
            'graph_kind': 'TEXT', 'activation_function': 'TEXT', 'qubits': 'INT', 'solution': 'TEXT',
            'unrounded_solution': 'TEXT', 'max_energy': 'FLOAT', 'min_energy': 'FLOAT', 'energy_ratio': 'FLOAT',
            'initial_parameters': 'TEXT', 'parameters': 'TEXT', 'number_parameters': 'INT', 'epochs': 'INT',
            'time': 'FLOAT'}
    unique = ['kind', 'instance', 'layer_number', 'nodes_number', 'optimization', 'compression', 'pauli_string_length',
              'entanglement', 'graph_kind', 'trial', 'activation_function']
    connect_database(name_database)
    create_table(name_database, name_database, rows, unique)


def get_exact_solution(name_database, name_table, data_to_read, parameters_to_fix):
    return read_data(name_database, name_table, data_to_read, parameters_to_fix)


def graph_to_dict(graph):
    adjacency_matrix = nx.to_numpy_array(graph)
    edges = {}
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i):
            if adjacency_matrix[i][j] == 0:
                continue
            edges[(i, j)] = adjacency_matrix[i][j]
    return edges
