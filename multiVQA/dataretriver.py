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
import math
import json


class Benchmarker(object):

    def __init__(self, kind, nodes_number, starting, ending, trials=1, layer_number=None, optimization='None',
                 initial_parameters='None', ratio_total_words='None', pauli_string_length='None', compression=None,
                 lower_order_terms=None,
                 entanglement='None',
                 graph_dict=None, graph_kind='indexed', activation_function='None', hyperparameters='None', shuffle=False, qubits=None, same_letter=True):

        if layer_number is None:
            layer_number = ['None']
        self.kind = kind
        self.nodes_number = nodes_number
        self.starting = starting
        self.ending = ending
        self.trials = trials
        self.layer_number = layer_number
        self.optimization = optimization
        self.initial_parameters = initial_parameters
        self.ratio_total_words = ratio_total_words
        self.pauli_string_length = pauli_string_length
        self.entanglement = entanglement
        self.graph_dict = graph_dict
        self.graph_kind = graph_kind
        self.activation_function = activation_function
        self.compression = compression
        self.lower_order_terms = lower_order_terms
        self.hyperparameters = hyperparameters
        self.shuffle = shuffle
        self.same_letter = same_letter
        self.qubits = qubits
        if self.compression is not None:
            if self.qubits is None:
                if self.lower_order_terms:
                    self.qubits = math.ceil(max(self.solve_quadratic(1, 1, -2 / 3 * self.nodes_number)))
                else:
                    self.qubits = math.ceil(max(self.solve_quadratic(1, -1, -2 / 3 * self.nodes_number)))
            else:
                self.qubits = qubits
            self.pauli_string_length = self.qubits
        if pauli_string_length != 'None':
            self.qubits = MultibaseVQA.get_num_qubits(self.nodes_number, self.pauli_string_length,
                                        self.ratio_total_words)

        if self.qubits is None or self.qubits < 15:
            qibo.set_backend("numpy")
            my_time = time()
            self._eigensolver_evaluater_parallel()
            print("Total time:", time() - my_time)
        else:
            qibo.set_backend("numpy")
            my_time = time()
            self._eigensolver_evaluater_serial()
            print("Total time:", time() - my_time)



    def _eigensolver_evaluater_parallel(self):
        process_number = 96
        pool = mp.Pool(process_number)
        if self.graph_dict is not None:
            [pool.apply_async(self._single_graph_evaluation, (instance, trial, (graph, self.graph_dict[graph]), layer)) for layer in
             self.layer_number for instance in
             range(self.starting, self.ending) for graph in self.graph_dict for trial in range(self.trials)]
        #     self.kind = 'bruteforce'
        #     [pool.apply_async(self._single_graph_evaluation, (instance, self.trials,
        #                                                       self.graph_dict[instance])) for instance in
        #      self.graph_dict if self.graph_dict[instance].number_of_nodes() < 20]
            # add the non-brute force
        [pool.apply_async(self._single_graph_evaluation, (instance, trial, self.graph_dict, layer)) for layer in self.layer_number for instance in
             range(self.starting, self.ending) for trial in range(self.trials)]
        pool.close()
        pool.join()

    def _eigensolver_evaluater_serial(self):
        if self.graph_dict is not None:
            for layer in self.layer_number:
                for instance in range(self.starting, self.ending):
                    for graph in self.graph_dict:
                        for trial in range(self.trials):
                            self._single_graph_evaluation(instance, trial, (graph, self.graph_dict[graph]), layer)
        else:
            for layer in self.layer_number:
                print(f'Layer number:{layer}')
                for trial in range(self.trials):
                    for instance in range(self.starting, self.ending):
                        self._single_graph_evaluation(instance, trial, self.graph_dict, layer)


    def _single_graph_evaluation(self, instance, trial, graph, layer):
        if graph is None:
            graph, instance = self._do_graph(instance)
            instance_name = str(instance)
        else:
            instance_name = graph[0]
            graph = graph[1]


        if self.kind == 'bruteforce':
            my_time = time()
            result = brute_force_graph(graph)
            timing = my_time - time()
            max_energy, min_energy, solution, energy_ratio = result[0], result[1], result[2], 1
            qubits, self.compression, self.pauli_string_length, epochs, parameters, number_parameters, unrounded_solution, initial_parameters, activation_function_name = 'None', 'None', 'None', 'None', 'None', 'None', 'None', self.initial_parameters, self.activation_function
        else:
            np.random.seed(trial)
            result_exact = self._get_exact_solution(instance)
            if self.kind == 'goemans_williamson':
                my_time = time()
                result = goemans_williamson(graph)
                timing = my_time - time()
                max_energy, solution = result[1], str(result[0])
                energy_ratio = (max_energy - result_exact[0][1]) / (result_exact[0][0] - result_exact[0][1])
                qubits, self.ratio_total_words, self.pauli_string_length, epochs, parameters, number_parameters, unrounded_solution, min_energy, initial_parameters, activation_function_name = 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', self.initial_parameters, self.activation_function
            else:
                adjacency_matrix, max_eigenvalue = self._graph_to_dict(graph)
                if self.kind == 'classicVQE':
                    self.pauli_string_length = 1
                    self.ratio_total_words = 1 / 3
                    self.activation_function = MultibaseVQA.linear_activation()

                qubits = self.qubits
                circuit = var_form(qubits, layer, self.entanglement)
                if self.initial_parameters == 'None':
                    initial_parameters = np.pi * np.random.uniform(0, 2, len(circuit.get_parameters(format='flatlist')))
                else:
                    initial_parameters = self._smart_initialization(instance, trial, circuit)

                solver = MultibaseVQA(circuit, adjacency_matrix, max_eigenvalue, hyperparameters=self.hyperparameters)
                if self.compression is not None:
                    solver.encode_nodes(self.nodes_number, self.pauli_string_length,
                                        compression=self.compression, lower_order_terms=self.lower_order_terms, shuffle=self.shuffle, seed=(trial+instance), same_letter=self.same_letter)
                else:
                    solver.encode_nodes(self.nodes_number, self.pauli_string_length, self.ratio_total_words)

                solver.set_activation(self.activation_function)

                my_time = time()
                result, cut, parameters, extra, unrounded_solution, solution = solver.minimize(initial_parameters,
                                                                                               method=self.optimization,
                                                                                               options={
                                                                                                   'maxiter': 1000000000})
                timing = time() - my_time

                print(timing)
                if self.optimization == 'cma':
                    epochs = extra[1].result[3]
                elif self.optimization != 'sgd':
                    epochs = extra['nfev']

                max_energy, min_energy, number_parameters, initial_parameters, parameters, unrounded_solution, solution = cut, 'None', len(
                    initial_parameters), str(initial_parameters.tolist()), str(parameters.tolist()), str(
                    unrounded_solution), str(solution)
                energy_ratio = (cut - result_exact[0][1]) / (result_exact[0][0] - result_exact[0][1])
                activation_function_name = self.activation_function.__name__
            if max_energy == energy_ratio:
                energy_ratio = 'None'

        instance = instance_name
        row = {'kind': self.kind, 'instance': str(instance), 'trial': trial, 'layer_number': layer,
               'nodes_number': self.nodes_number, 'optimization': self.optimization,
               'activation_function': str(activation_function_name),
               'compression': self.ratio_total_words, 'pauli_string_length': self.pauli_string_length,
               'entanglement': self.entanglement,
               'graph_kind': self.graph_kind, 'qubits': qubits, 'solution': solution,
               'unrounded_solution': unrounded_solution,
               'max_energy': max_energy, 'min_energy': min_energy, 'energy_ratio': energy_ratio,
               'initial_parameters': initial_parameters, 'parameters': parameters,
               'number_parameters': number_parameters, 'hyperparameter': str(self.hyperparameters),
               'epochs': epochs, 'time': timing}
        insert_value_table('MaxCutDatabase', 'MaxCutDatabase', row)


    def _graph_to_dict(self, graph):
        eigenvalues, _ = np.linalg.eig(nx.to_numpy_matrix(graph))
        max_eigenvalue = np.max(eigenvalues)
        min_eigenvalue = np.min(eigenvalues)
        adjacency_matrix = nx.to_numpy_array(graph)
        edges = {}
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i):
                if adjacency_matrix[i][j] == 0:
                    continue
                edges[(i, j)] = adjacency_matrix[i][j]
        return edges, max_eigenvalue #(max_eigenvalue+min_eigenvalue)/2

    def _do_graph(self, instance):
        true_random_graphs = False
        fully_connected = False
        if self.graph_kind == 'random':
            true_random_graphs = True
        if self.graph_kind == 'fully':
            fully_connected = True
        graph = RandomGraphs(instance, self.nodes_number, true_random_graphs, fully_connected).graph
        if true_random_graphs:
            instance = graph.return_index()
        return graph, instance

    def _get_exact_solution(self,instance):
        result_exact = read_data('MaxCutDatabase', 'MaxCutDatabase', ['max_energy', 'min_energy'],
                                                {'kind': 'bruteforce', 'instance': instance,
                                                 'nodes_number': self.nodes_number,
                                                 'graph_kind': self.graph_kind})
        if len(result_exact) == 0:
            result_exact = [(1, 0)]
        return result_exact

    def _smart_initialization(self, instance, trial, circuit, layer):
        if layer == 0:
            return np.pi * np.random.uniform(0, 2, len(circuit.get_parameters(format='flatlist')))
        else:
            previous_parameters = read_data('MaxCutDatabase', 'MaxCutDatabase', ['parameters'],
                                                { 'instance': instance, 'trial':trial, 'layer_number': f'{int(layer)-1}'})
            np.random.seed(layer*instance*trial)
            previous_parameters = np.array([json.loads(previous_parameters[j][0]) for j in range(len(previous_parameters))])
            added_parameters = np.pi * np.random.uniform(0, 2, len(circuit.get_parameters(format='flatlist'))- len(previous_parameters[0]))
        return np.append(previous_parameters, added_parameters)

    @staticmethod
    def initialize_database(name_database):
        rows = {'kind': 'TEXT', 'instance': 'TEXT', 'trial': 'INT', 'layer_number': 'INT', 'nodes_number': 'INT',
                'optimization': 'TEXT', 'compression': 'FLOAT', 'pauli_string_length': 'INT', 'entanglement': 'TEXT',
                'graph_kind': 'TEXT', 'activation_function': 'TEXT', 'qubits': 'INT', 'solution': 'TEXT',
                'unrounded_solution': 'TEXT', 'max_energy': 'FLOAT', 'min_energy': 'FLOAT', 'energy_ratio': 'FLOAT',
                'initial_parameters': 'TEXT', 'parameters': 'TEXT', 'number_parameters': 'INT', 'hyperparameter': 'TEXT','epochs': 'INT',
                'time': 'FLOAT'}
        unique = ['kind', 'instance', 'layer_number', 'nodes_number', 'optimization', 'compression','hyperparameter',
                  'pauli_string_length',
                  'entanglement', 'graph_kind', 'trial', 'activation_function']
        connect_database(name_database)
        create_table(name_database, name_database, rows, unique)

    @staticmethod
    def nodes_compressed(quibits):
        return int((3 * (quibits ** 2 + quibits) / 2))

    @staticmethod
    def max_compression(quibits):
        return 4 ** quibits - 1

    @staticmethod
    def solve_quadratic(a, b, c):
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            x_1 = (-b+math.sqrt(discriminant))/2*a
            x_2 = (-b-math.sqrt(discriminant))/2*a
        else:
            x_1 = complex((-b/(2*a)), math.sqrt(-discriminant)/(2*a))
            x_2 = complex((-b/(2*a)), -math.sqrt(-discriminant)/(2*a))
        return x_1, x_2

