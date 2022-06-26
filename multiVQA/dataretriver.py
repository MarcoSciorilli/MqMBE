from typing import Tuple
from multiVQA.newgraph import RandomGraphs
from multiVQA.resultevaluater import brute_force_graph, goemans_williamson
from multiVQA.ansatz import var_form
from multiVQA.multibase import MultibaseVQA
from multiVQA.datamanager import insert_value_table, connect_database, create_table, read_data
import math
import json
import MQLib
import qibo
import networkx as nx
import multiprocessing as mp
import numpy as np
from time import time


class Benchmarker(object):

    def __init__(self, kind: str = 'multibaseVQA', graph_dict: dict = None, nodes_number: int = 18, starting: int = 0,
                 ending: int = 100, trials: int = 1, layer_number: list = None,
                 optimization: str = 'None',
                 initial_parameters: str = 'None', ratio_total_words: float = 'None', pauli_string_length: int = 'None',
                 compression: int = None,
                 lower_order_terms: bool = None,
                 entanglement: str = 'None',
                 graph_kind: str = 'indexed', activation_function: callable = 'None', hyperparameters: list = 'None',
                 shuffle: bool = False, qubits: int = None, same_letter: bool = True, precision: str = 'double',
                 database_name: str = 'MaxCutDatabase',
                 warmup: bool = False, multiprocessing: bool = True) -> None:
        """
        Demo function to benchmark the multiqubit multi-basis encoding. It gets from input a series of requirement from
        the user concerning the benchmark itself, and save the results in a very naive SQLite table.
        :param kind: Name of the time of algorithm used (if classical, give the MQlib code), default is MqMBE
        :param nodes_number: Number of nodes in the graph to cut
        :param graph_dict: Dictionary containing the graph instances to cut. The instances must be pairs of key (strings)
                            giving a name to the instance, and item (a nx.graph). Is no dictionary is provided by the user
                            the graphs are generated with random density and random integer weights between -10 and 10,
                            according to the following parameters.
        :param starting: Starting point for the generation of random instances. A instance will be generated for every
                        integer between 'starting' and 'ending'. The integer number assigned to every instance works
                        both as seed for the generation and as name for the instance.
        :param ending: Ending point for the generation of the random instances.
        :param trials: Number of different random initialisation of the parameters of the circuit to try for every
                        instance.
        :param layer_number: List containing the number of layers to use in the circuit.
        :param optimization: Classical optimize to use in the VQA. Scipy optimization and cma are supported, keras one
                            still needs some work.
        :param initial_parameters:Whether or no reuse the parameters found in the previous training as a starting point
                                    for a new optimization. Works only if all the layers are trained subsequently.
        :param pauli_string_length: maximum length of the pauli words used in the compression.
        :param ratio_total_words: ratio of the pauli word to use in the compression among all the feasible ones up to
                                pauli_string_length
        :param compression: Order of the compression to be used (if explicitly expressed)
        :param lower_order_terms: Whether or no also include shorter pauli word.
        :param entanglement: Architecture to be used in the circuit (name to be changed)
        :param graph_kind:  If randomly generate, which kind of graph ('fully' for fully connected, 'indexed' for
                            the ones totally random generated from their integer index, 'random' for the ones totally
                            random, so not generated from their index)
        :param activation_function: Activation function to be used in the loss function.
        :param hyperparameters: Hyperparameters to be used in the loss function.
        :param shuffle: Whether of not use random observables.
        :param qubits: Number of qubits to be used in the circuit.
        :param same_letter: Whether of not use observables of the same pauli letter.
        :param precision: Whether use single or doable precision in the simulation of the circuits.
        :param database_name: Name of the database (table) in which all the output data should be stored.
        :param warmup: Whether or no perform some warmup on the circuit.
        :param multiprocessing: Whether or no use multiprocessing.
        """
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
        self.precision = precision
        self.database_name = database_name
        self.warmup = warmup

        # If the compression rate is given, find the required number of qubits
        if self.compression is not None:
            if self.qubits is None:
                if self.lower_order_terms:
                    self.qubits = math.ceil(max(self.solve_quadratic(1, 1, -2 / 3 * self.nodes_number)))
                else:
                    self.qubits = math.ceil(max(self.solve_quadratic(1, -1, -2 / 3 * self.nodes_number)))
            else:
                self.qubits = qubits
            self.pauli_string_length = self.qubits
        # Else, infer it from the given parameters
        else:
            self.qubits = MultibaseVQA.get_num_qubits(self.nodes_number, self.pauli_string_length,
                                                      self.ratio_total_words)

        # Run the process in serial or in parallel depending on the user choice
        if (self.qubits is None or self.qubits < 25) and multiprocessing is True:
            qibo.set_backend("numpy")
            qibo.set_precision(precision)
            my_time = time()
            self._eigensolver_evaluater_parallel()
            print("Total time:", time() - my_time)
        else:
            qibo.set_backend("numpy")
            qibo.set_precision(precision)
            my_time = time()
            self._eigensolver_evaluater_serial()
            print("Total time:", time() - my_time)

    def _eigensolver_evaluater_parallel(self) -> None:
        """
        Function which carry on the benchmark in a multiprocess framework, running an instance per process
        """
        process_number = 64
        pool = mp.Pool(process_number)
        if self.graph_dict is not None:
            [pool.apply_async(self._single_graph_evaluation, (0, trial, (graph, self.graph_dict[graph]), layer)) for
             layer in
             self.layer_number for graph in self.graph_dict for trial in range(self.trials)]
        #     self.kind = 'bruteforce'
        #     [pool.apply_async(self._single_graph_evaluation, (instance, self.trials,
        #                                                       self.graph_dict[instance])) for instance in
        #      self.graph_dict if self.graph_dict[instance].number_of_nodes() < 20]
        # add the non-brute force
        [pool.apply_async(self._single_graph_evaluation, (instance, trial, self.graph_dict, layer)) for layer in
         self.layer_number for instance in
         range(self.starting, self.ending) for trial in range(self.trials)]
        pool.close()
        pool.join()

    def _eigensolver_evaluater_serial(self) -> None:
        """
        Function which carry on the benchmark in a serial framework, running one instance at a time
        """
        if self.graph_dict is not None:
            for layer in self.layer_number:
                for graph in self.graph_dict:
                    for trial in range(self.trials):
                        self._single_graph_evaluation(0, trial, (graph, self.graph_dict[graph]), layer)
        else:
            for layer in self.layer_number:
                print(f'Layer number:{layer}')
                for trial in range(self.trials):
                    for instance in range(self.starting, self.ending):
                        self._single_graph_evaluation(instance, trial, self.graph_dict, layer)

    def _single_graph_evaluation(self, instance: int, trial: int, graph: tuple, layer: int) -> None:
        """
        Function that run the benchmark of a single instance.
        :param instance: If generated, the index of the instance.
        :param trial: The index of the trial on the instance.
        :param graph: If given, a tuple with the instance name and the instance itself as a nx.graph.
        :param layer: Number of layer to be used in the circuit.
        """
        # If the graph is not given, create it
        if graph is None:
            graph, instance = self._do_graph(instance)
            instance_name = str(instance)
        else:
            instance_name = graph[0]
            graph = graph[1]

        # Use bruteforce to find the maxcut
        if self.kind == 'bruteforce':
            # Measure runtime and get the result
            my_time = time()
            result = brute_force_graph(graph)
            timing = my_time - time()
            # Fix all the features of the dataset
            max_energy, min_energy, solution, energy_ratio = result[0], result[1], result[2], 1
            qubits, self.compression, self.pauli_string_length, epochs, parameters, number_parameters, unrounded_solution, initial_parameters, activation_function_name = 'None', 'None', 'None', 'None', 'None', 'None', 'None', self.initial_parameters, self.activation_function
        else:
            # If using a non-deterministic algorithm, initialise the seed
            np.random.seed(trial + instance)
            # Retrieve the right solution
            result_exact = self._get_exact_solution(instance, graph)
            # If Goemans williamsons is required, use the internal function
            if self.kind == 'goemans_williamson':
                my_time = time()
                result = goemans_williamson(graph)
                timing = my_time - time()
                max_energy, solution = result[1], str(result[0])
                energy_ratio = (max_energy / (result_exact[0][0]))
                qubits, self.ratio_total_words, self.pauli_string_length, epochs, parameters, number_parameters, unrounded_solution, min_energy, initial_parameters, activation_function_name = 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', self.initial_parameters, self.activation_function
            elif self.kind == 'classicVQE' or self.kind == 'multibaseVQA':
                # Shape the adjacency matrix to optimize the computation (also dependent on machine precision)
                if self.precision == 'single':
                    # To change the type of the graph
                    adjacency_matrix, max_eigenvalue = self._graph_to_dict(graph)
                else:
                    adjacency_matrix, max_eigenvalue = self._graph_to_dict(graph)

                # If the classic VQE is required, fix the parameters to get the right encoding
                if self.kind == 'classicVQE':
                    self.pauli_string_length = 1
                    self.ratio_total_words = 1 / 3
                    self.activation_function = MultibaseVQA.linear_activation
                # Create the circuits with the wanted initialization of the parameters
                qubits = self.qubits
                circuit = var_form(qubits, layer, self.entanglement)
                if self.initial_parameters == 'None':
                    initial_parameters = np.pi * np.random.uniform(-1, 1,
                                                                   len(circuit.get_parameters(format='flatlist')))
                else:
                    initial_parameters = self._smart_initialization(instance, trial, circuit)

                if self.precision == 'single':
                    initial_parameters = np.float32(initial_parameters)
                    self.hyperparameters = np.float32(self.hyperparameters)

                # Initialize the encoding
                solver = MultibaseVQA(circuit, adjacency_matrix, max_eigenvalue, hyperparameters=self.hyperparameters)
                if self.compression is not None:
                    solver.encode_nodes(self.nodes_number, self.pauli_string_length,
                                        compression=self.compression, lower_order_terms=self.lower_order_terms,
                                        shuffle=self.shuffle, seed=(trial + instance), same_letter=self.same_letter)
                else:
                    solver.encode_nodes(self.nodes_number, self.pauli_string_length, self.ratio_total_words)
                # Set the activation function
                solver.set_activation(self.activation_function)

                # Fix some bounds for bounded optimizer (only COBYLA for the moment)
                if self.optimization == 'COBYLA':
                    bounds = [[-np.pi, np.pi] for i in range(len(initial_parameters))]
                    cons = []
                    for factor in range(len(bounds)):
                        lower, upper = bounds[factor]
                        l = {'type': 'ineq',
                             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
                        u = {'type': 'ineq',
                             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
                        cons.append(l)
                        cons.append(u)
                    bounds = None
                else:
                    bounds = [(-np.pi, np.pi) for i in range(len(initial_parameters))]
                    cons = None

                # If warmup is required, do it (requires a database containing the starting solution) ####TO FIX####
                if self.warmup:
                    import ast
                    solution = read_data('MaxCutDatabase_512_bur', 'MaxCutDatabase_512_bur', ['solution'],
                                         {'instance': instance_name, 'trial': trial})
                    solution = solution[0][0]
                    solution = solution[:1] + solution[(1 + 1):]
                    solution = str(solution).replace('  ', ',')
                    solution = str(solution).replace(' ', ',')
                    solution = ast.literal_eval(solution)

                    # solution = np.array([json.loads(solution[0][0])])
                    solver.set_approx_solution(solution)
                    my_time = time()
                    result, cut, parameters, extra, unrounded_solution, solution = solver.minimize(initial_parameters,
                                                                                                   method=self.optimization,
                                                                                                   bounds=bounds,
                                                                                                   constraints=cons,
                                                                                                   tol=0.1,
                                                                                                   options={
                                                                                                       'maxiter': 1000000000},
                                                                                                   warmup=self.warmup)
                    initial_parameters = parameters
                    timing = time() - my_time
                    print(f'Warmup time:{timing}')
                # Start the timed optimization
                my_time = time()
                result, cut, parameters, extra, unrounded_solution, solution = solver.minimize(initial_parameters,
                                                                                               method=self.optimization,
                                                                                               bounds=bounds, tol=1e-03,
                                                                                               constraints=cons,
                                                                                               options={
                                                                                                   'maxiter': 1000000000})
                timing = time() - my_time
                print(timing)
                # Fix the results shape depending on the optimizer used
                if self.optimization == 'cma':
                    epochs = extra[1].result[3]
                elif self.optimization != 'sgd':
                    epochs = extra['nfev']
                else:
                    epochs = extra
                # Fix all the features of the dataset
                max_energy, min_energy, number_parameters, initial_parameters, parameters, unrounded_solution, solution = cut, 'None', len(
                    initial_parameters), str(initial_parameters.tolist()), str(parameters.tolist()), str(
                    unrounded_solution), str(solution)
                energy_ratio = (cut) / (result_exact[0][0])
                activation_function_name = self.activation_function.__name__
            else:
                # Use a classic heuristic from MQLib to perform the optimization
                instance = MQLib.Instance('M', nx.to_numpy_array(graph))
                result = MQLib.runHeuristic(self.kind, instance, 0.00001, seed=trial)
                timing = result['bestsolhistory_runtimes'][1]
                max_energy, solution = result['objval'], str(result['solution'])
                energy_ratio = (max_energy) / (result_exact[0][0])
                qubits, self.ratio_total_words, self.pauli_string_length, epochs, parameters, number_parameters, unrounded_solution, min_energy, initial_parameters, activation_function_name = 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', self.initial_parameters, self.activation_function

            if max_energy == energy_ratio:
                energy_ratio = 'None'

        # Save the instances in the dataset
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
        insert_value_table(self.database_name, self.database_name, row)

    @staticmethod
    def _graph_to_dict(graph: object) -> Tuple[dict, np.array]:
        """
        Function that, given a graph, return a dictionary of the edges and the maximum eigenvalue of the
        adjacency matrix.
        :param graph: A nx graph.
        :return: A dictionary of the edges and the maximum eigenvalue of the adjacency matrix.
        """
        adj_matrix = nx.to_numpy_array(graph)
        eigenvalues, _ = np.linalg.eig(adj_matrix)
        max_eigenvalue = np.max(eigenvalues)
        # max_eigenvalue = np.sum(np.abs(eigenvalues))
        edges = {}
        for i in range(adj_matrix.shape[0]):
            for j in range(i):
                if adj_matrix[i][j] == 0:
                    continue
                edges[(i, j)] = adj_matrix[i][j]
        return edges, max_eigenvalue  # (max_eigenvalue+min_eigenvalue)/2

# Single precision version of the preparation of the graph ##### TO FIX #####
    @staticmethod
    def _graph_to_dict_32(graph: nx.graph) -> Tuple[dict, np.array]:
        adj_matrix = np.zeros(shape=[int(max(graph.adj)) + 1, int(max(graph.adj)) + 1])
        for i in graph.adj:
            for j in graph.adj[i]:
                if graph.adj[i][j]['weight'] == 0:
                    continue
                adj_matrix[int(i) - 1, int(j) - 1] = graph.adj[i][j]['weight']
        eigenvalues, _ = np.linalg.eig(adj_matrix)
        max_eigenvalue = np.max(eigenvalues)
        edges = {}
        for i in range(adj_matrix.shape[0]):
            for j in range(i):
                if adj_matrix[i][j] == 0:
                    continue
                edges[(np.int32(i), np.int32(j))] = np.float32(adj_matrix[i][j])
        return edges, np.float32(max_eigenvalue)  # (max_eigenvalue+min_eigenvalue)/2

    def _do_graph(self, instance: int) -> Tuple[object, int]:
        """
        Function which, given the user specification, return the wanted generated graph
        :param instance: Index of the graph to generate.
        :return: The graph.
        """
        true_random_graphs = False
        fully_connected = False
        if self.graph_kind == 'random':
            true_random_graphs = True
        if self.graph_kind == 'fully':
            fully_connected = True
        graph = RandomGraphs(instance, self.nodes_number, true_random_graphs, fully_connected=fully_connected).graph
        if true_random_graphs:
            instance = graph.return_index()
        return graph, instance

    def _get_exact_solution(self, instance: int, graph: object) -> list:
        """
        Function that, given a graph, return either the exact maxcut solution, or the one found by BURER heuristic
        :param instance: istance index number
        :param graph: graph to find the maxcut
        :return:
        """
        result_exact = read_data(self.database_name, self.database_name, ['max_energy', 'min_energy'],
                                 {'kind': 'bruteforce', 'instance': instance,
                                  'nodes_number': self.nodes_number,
                                  'graph_kind': self.graph_kind})
        if len(result_exact) == 0:
            instance = MQLib.Instance('M', nx.to_numpy_array(graph))
            result = MQLib.runHeuristic('BURER2002', instance, 0.5)
            max_energy = result['objval']
            result_exact = [(max_energy, 0)]
        return result_exact

    def _smart_initialization(self, instance: int, trial: int, circuit: qibo.models.Circuit, layer: int) -> np.array:
        """
        Function which initialize the parameters of a circuit with the ones found by the optimization of the circuit
        with one less layer.
        :param instance: Index of the specific instance.
        :param trial: Index of the specific trial.
        :param circuit: Circuit architecture
        :param layer: Number of layers in the circuits.
        :return: Array of the parameters for the new initialization.
        """
        if layer == 0:
            return np.pi * np.random.uniform(0, 2, len(circuit.get_parameters(format='flatlist')))
        else:
            previous_parameters = read_data(self.database_name, self.database_name, ['parameters'],
                                            {'instance': instance, 'trial': trial, 'layer_number': f'{int(layer) - 1}'})
            np.random.seed(layer * instance * trial)
            previous_parameters = np.array(
                [json.loads(previous_parameters[j][0]) for j in range(len(previous_parameters))])
            added_parameters = np.pi * np.random.uniform(0, 2, len(circuit.get_parameters(format='flatlist')) - len(
                previous_parameters[0]))
        return np.append(previous_parameters, added_parameters)

    @staticmethod
    def initialize_database(name_database: str) -> None:
        """
        Function that initialize the database for the benchmark, with a user defined name.
        :param name_database: Name of the database
        """
        rows = {'kind': 'TEXT', 'instance': 'TEXT', 'trial': 'INT', 'layer_number': 'INT', 'nodes_number': 'INT',
                'optimization': 'TEXT', 'compression': 'FLOAT', 'pauli_string_length': 'INT', 'entanglement': 'TEXT',
                'graph_kind': 'TEXT', 'activation_function': 'TEXT', 'qubits': 'INT', 'solution': 'TEXT',
                'unrounded_solution': 'TEXT', 'max_energy': 'FLOAT', 'min_energy': 'FLOAT', 'energy_ratio': 'FLOAT',
                'initial_parameters': 'TEXT', 'parameters': 'TEXT', 'number_parameters': 'INT',
                'hyperparameter': 'TEXT', 'epochs': 'INT',
                'time': 'FLOAT'}
        unique = ['kind', 'instance', 'layer_number', 'nodes_number', 'optimization', 'compression', 'hyperparameter',
                  'pauli_string_length',
                  'entanglement', 'graph_kind', 'trial', 'activation_function']
        connect_database(name_database)
        create_table(name_database, name_database, rows, unique)

    @staticmethod
    def nodes_compressed(qubits: int) -> int:
        return int((3 * (qubits ** 2 + qubits) / 2))

    @staticmethod
    def max_compression(qubits: int) -> int:
        return 4 ** qubits - 1

    @staticmethod
    def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            x_1 = (-b + math.sqrt(discriminant)) / 2 * a
            x_2 = (-b - math.sqrt(discriminant)) / 2 * a
        else:
            x_1 = complex((-b / (2 * a)), math.sqrt(-discriminant) / (2 * a))
            x_2 = complex((-b / (2 * a)), -math.sqrt(-discriminant) / (2 * a))
        return x_1, x_2
