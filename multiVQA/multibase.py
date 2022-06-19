import collections
import math
import copy
import qibo.models
from numpy import ceil, floor
from qibo.symbols import I, X, Y, Z
from qibo.config import raise_error
from qibo import K
import tensorflow as tf
import numpy as np
import random
from itertools import combinations, product


class MultibaseVQA(object):
    from qibo import optimizers

    def __init__(self, circuit: qibo.models.Circuit, adjacency_matrix: np.array, max_eigenvalue: np.array,
                 hyperparameters: np.array, expectations_method=False) -> None:
        """
        Initialization of the class
        :param circuit: A Qibo circuits.
        :param adjacency_matrix: Adjacency matrix of the graph to cut.
        :param max_eigenvalue: Maximum eigenvalue of the Adjacency matrix of the graph.
        :param hyperparameters: hyperparameters used in the loss function
        :type expectations_method: Switch to choose how to evaluate the expectations values
        """
        # Initialisation of the class
        self.activation_function = self.linear_activation
        self.circuit = circuit
        self.adjacency_matrix = adjacency_matrix
        self.parameter_iteration = []
        self.max_eigenvalue = max_eigenvalue
        self.hyperparameters = hyperparameters
        self.approx_solution = None
        self.node_expectation_mapping = 0
        self.expectations_method = expectations_method

    @staticmethod
    def get_num_qubits(num_nodes: int, pauli_string_length: int, ratio_total_words: float) -> int:
        """
        Function which, given the number of nodes of the graph problem, and the configurations of the compression,
        returns the number of qubits necessary to carry on the algorithm
        :param num_nodes: number of nodes in the graph.
        :param pauli_string_length: maximum length of the pauli words used in the compression.
        :param ratio_total_words: ratio of the pauli word to use in the compression among all the feasable ones up to
                                pauli_string_length
        :return: number of qubits needed in the circuit
        """
        # return the number of qubits necessary
        return int(ceil(num_nodes / round((4 ** pauli_string_length - 1) * ratio_total_words)) * pauli_string_length)

    @staticmethod
    def linear_activation(x: float) -> float:
        """
        Basic activation function
        :param x: expectation value
        :return: same value
        """
        return x

    @staticmethod
    def _pauli_string_same_letter(pauli_string_length: int, order: int, lower_order_terms: bool, shuffle: bool,
                                  seed: int,
                                  pauli_letters: int = 4) -> list:
        """
        Function which returns a list of all the pauli words satisfying the requirements asked by the user.
        :param pauli_string_length: Maximum length of any pauli word (also considering Identities).
        :param order: Maximum number of qubits actually interested in the pauli word (closely dependent on the compression).
        :param lower_order_terms: Whether or no also include shorter pauli word.
        :param shuffle: Whether or no shuffle the list of pauli word.
        :param seed: Seed for the shuffling.
        :param pauli_letters:Number of pauli letters to be used in the pauli words (obv max 4).
        :return: list of pauli word encoded as tuples of integers.
        """
        # Initialise the list
        pauli_string = []

        # If required, use all pauli words of length up to order
        if lower_order_terms:
            smallest_length = 1
        else:
            smallest_length = order

        # Append all the combinations of given length of same pauli letter
        for i in range(1, pauli_letters):
            for k in range(smallest_length, order + 1):
                comb = combinations(list(range(pauli_string_length)), k)
                for positions in comb:
                    instance = [0] * pauli_string_length
                    for index in positions:
                        instance[index] = i
                    pauli_string.append(tuple(instance))

        # If required, shuffle the list
        if shuffle:
            random.seed(seed)
            random.shuffle(pauli_string)
        return pauli_string

    @staticmethod
    def _random_pauli_string(pauli_string_length: int, order: int, lower_order_terms: bool, shuffle: bool,
                             seed: int) -> list:
        """
        Same as _pauli_string_same_letter, but picking instead random pauli words among, not only the one of the
        same letter.
        :param pauli_string_length: Maximum length of any pauli word (also considering Identities).
        :param order: Maximum number of qubits actually interested in the pauli word (closely dependent on
                    the compression).
        :param lower_order_terms: Whether or no also include shorter pauli word.
        :param shuffle: Whether or no shuffle the list of pauli word.
        :param seed: Seed for the shuffling.
        :return:
        """
        # Initialise all possibile tuples (pauli letter, qubit)
        pauli_tuples = [(i, j) for i in range(1, 4) for j in range(pauli_string_length)]

        # If required, use all pauli words of length up to order
        if lower_order_terms:
            smallest_length = 1
        else:
            smallest_length = order

        # Create list of all possible combinations of pauli word of given length
        total_combinations = []
        for i in range(smallest_length, order + 1):
            total_combinations = total_combinations + (list(combinations(pauli_tuples, i)))

        # Create the final list of pauli words
        pauli_string = []
        for comb in total_combinations:
            instance = [0] * pauli_string_length
            for j in comb:
                instance[j[1]] = j[0]
                pauli_string.append(instance)

        # If required, shuffle the list
        if shuffle:
            random.seed(seed)
            random.shuffle(pauli_string)
        return pauli_string

    def _callback(self, x: int) -> None:
        """
        Callback function to record the number of call of the loss function
        :param x:
        """
        self.parameter_iteration.append(x)

    def encode_nodes(self, num_nodes: int, pauli_string_length: int, ratio_total_words: float = None,
                     compression: int = None,
                     lower_order_terms: bool = False, shuffle: bool = True, seed: int = 0,
                     same_letter: bool = True) -> int:
        """
        Function which save the encodings of the graph nodes in the chosen observables (expressed in symbolic Hamiltonians)
        :param num_nodes: number of nodes in the graph.
        :param pauli_string_length: Maximum length of any pauli word (also considering Identities).
        :param ratio_total_words: ratio of the pauli word to use in the compression among all the feasable ones up to
                                pauli_string_length
        :param compression: Order of the compression to be used (if explicitly expressed)
        :param lower_order_terms: Whether or no also include shorter pauli word.
        :param shuffle: Whether or no shuffle the list of pauli word.
        :param seed: Seed for the shuffling.
        :param same_letter: Whether of no use observables of the same pauli letter
        :return:
        """

        def get_pauli_word(indices: collections.Iterable, k: object, qubits: int = None) -> qibo.hamiltonians.SymbolicHamiltonian:
            """
            Function which, given a pauli word, return the corresponding symbolic hamiltonian
            :param indices: pauli word as a list of integers
            :param k: nober of the qubit of the observables
            :param qubits:total number of qubits used
            :return: Symbolic hamiltonian of the pauli word
            """
            from qibo import hamiltonians

            # Alternative method to encode the nodes, non-symbolic
            if self.expectations_method:
                pauli_matrices = ['I', 'X', 'Y', 'Z']
                word = tuple()
                for i in indices:
                    word += (pauli_matrices[i],)
                if qubits:
                    qubits_list = list(range(qubits))
                    qubits_list.remove(int(k))
                    for j in qubits_list:
                        word *= pauli_matrices[0](j)
                return word
            else:
                # Given the pauli word, write it as a symbolic hamiltonian
                # Generate pauli string corresponding to indices
                # where (0, 1, 2, 3) -> 1XYZ and so on
                pauli_matrices = np.array([I, X, Y, Z])
                word = np.int(1)
                for qubit, i in enumerate(indices):
                    word *= pauli_matrices[i](qubit + int(k))

                # If the number of qubits is given, add identity of all the unused qubits
                if qubits:
                    qubits_list = list(range(qubits))
                    qubits_list.remove(int(k))
                    for j in qubits_list:
                        word *= pauli_matrices[0](j)
                return hamiltonians.SymbolicHamiltonian(word)

        # If the compression is not explicity expressed, infer the encoding from pauli_string_length and ratio_total_words
        if compression is None:
            pauli_strings = self._pauli_string_same_letter(pauli_string_length, 1, True, shuffle, seed,
                                                           pauli_letters=int(ratio_total_words * 3 + 1))
            num_strings = len(pauli_strings)

            # position i stores string corresponding to the i-th node.
            self.node_mapping = [
                get_pauli_word(pauli_strings[int(i % num_strings)], pauli_string_length * floor(i / num_strings),
                               self.get_num_qubits(num_nodes, pauli_string_length, ratio_total_words)) for i in range(num_nodes)]
        else:
            # If the compression is explicitly expressed, get the list of the pauli words (subject to user requirements)
            if same_letter:
                pauli_strings = self._pauli_string_same_letter(pauli_string_length, compression, lower_order_terms,
                                                               shuffle, seed)
            else:
                pauli_strings = self._random_pauli_string(pauli_string_length, compression, lower_order_terms,
                                                          shuffle, seed)
            num_strings = len(pauli_strings)

            # Alternative method to express the expectation values (not important)
            if self.expectations_method:
                node_mapping = []
                for current_string in product(["I", "X", "Y", "Z"], repeat=9):
                    current_string_dic = {}
                    for j in current_string:
                        if j in current_string_dic:
                            current_string_dic[j] = current_string_dic[j] + 1
                        else:
                            current_string_dic[j] = 0
                    if len(current_string_dic) > 2 or 'I' not in current_string_dic or current_string_dic['I'] < 9 - 3 \
                            or current_string_dic['I'] > 9 - 3:
                        continue
                    else:
                        node_mapping.append(current_string)

                # Minimize the hamming distance between one observable and the next
                import scipy
                for i in range(len(node_mapping) - 1):
                    smallest = 1000
                    for j in range(len(node_mapping) - 1, i - 1, -1):
                        if scipy.spatial.distance.hamming(node_mapping[i], node_mapping[j]) < smallest:
                            smallest = j
                    node_mapping[i + 1], node_mapping[smallest] = node_mapping[smallest], node_mapping[i + 1]
                self.node_mapping = node_mapping
            else:

                # position i stores string corresponding to the i-th node.
                self.node_mapping = [
                    get_pauli_word(pauli_strings[int(i % num_strings)], pauli_string_length * floor(i / num_strings))
                    for i in range(num_nodes)]
        return ceil(num_nodes / num_strings)

    def set_activation(self, function: object) -> None:
        """
        Method to set the activation function used in the loss function
        :param function: the activation function
        """
        self.activation_function = function

    def set_circuit(self, circuit: qibo.models.Circuit) -> None:
        """
        Method to set the circuit used for the VQA
        :param circuit:A QIBO circuit 
        """
        self.circuit = circuit

    def set_approx_solution(self, solution: list) -> None:
        """
        Method to set an approximate solution to the problem used in the warmup fase of the algorithm (if used)
        :param solution:
        """
        self.approx_solution = solution

    def minimize(self, initial_state: list, method: str = 'Powell', jac: callable = None, hess = None,
                 hessp = None, bounds: list = None, constraints = (), tol: float = None, callback: object = None,
                 options: dict = None, processes: int = None, compile: bool = False, warmup: bool = False) -> object:
        """
        Function that carries on the minimization of the loss function
        :param initial_state: list of the starting values of the parameters of the circuit
        :param method: scipy method used to minimize the loss function
        :param jac: function that evaluate the gradient of the loss function
        :param hess: refert to qibo documentations for all the parameters not specified
        :param hessp:
        :param bounds:
        :param constraints:
        :param tol:
        :param callback:
        :param options:
        :param processes:
        :param compile:
        :param warmup: Whether or no warm up the circuit with a given solution
        :return: Multiple metrics on the found solution
        """
        def _loss(params: list, circuit: qibo.models.Circuit, adjacency_matrix: np.array, activation_function: callable, node_mapping: np.array) -> float:
            """
            Function that evaluate the loss function
            :param params: parameters used in the circuit
            :param circuit: qibo circuit used for the VQA
            :param adjacency_matrix: adjacency matrix of the graph to cut
            :param activation_function: activation function used in the loss
            :param node_mapping: encoding of the graph nodes in the observables
            :return: value of the loss
            """
            # defines loss function with given activation function
            circuit.set_parameters(params)
            qubits = circuit.nqubits

            loss = 0
            # Once again, different way to get the expectation values
            if self.expectations_method:
                final_state = circuit().numpy()
                tstate = np.copy(final_state)
                representation = []
                for gate in node_mapping:
                    tstate = gate(tstate)
                    representation.append(np.conj(tstate).dot(final_state).real)
                pauli_basis_representation = asarray(representation)
                node_mapping_expectation = pauli_basis_representation
            else:
                # Get the final state of the circuit and the expectation values
                final_state = circuit()
                node_mapping_expectation = [i.expectation(final_state) for i in node_mapping]
            self.node_expectation_mapping = node_mapping_expectation

            # Evaluate the first part of the loss function
            for i in adjacency_matrix:
                loss += adjacency_matrix[i] * activation_function(
                    node_mapping_expectation[i[0]] * self.hyperparameters[0] * qubits) \
                        * activation_function(node_mapping_expectation[i[1]] * self.hyperparameters[0] * qubits)

            # Evaluate the penalization term
            penalization = 0
            for i in range(len(node_mapping)):
                penalization += ((node_mapping_expectation[i]) ** 2)
            return loss + self.hyperparameters[1] * self.max_eigenvalue * (len(node_mapping) / 3 - 2 / 3) * penalization

        def _loss_gradient(params: list, circuit: qibo.models.Circuit, adjacency_matrix: np.array, activation_function: callable, node_mapping: np.array) -> list:
            """
            Function which evaluate the gradient of the loss function, assuming that tanh was the activation function,
            using parameter-shift rule.
            :param params: parameters used in the circuit
            :param circuit: qibo circuit used for the VQA
            :param adjacency_matrix: adjaceny matrix of the graph to cut
            :param activation_function: activation function used in the loss
            :param node_mapping: encoding of the graph nodes in the observables
            :return: list representing the gradient of the loss function
            """
            circuit.set_parameters(params)
            final_state = circuit()
            qubits = circuit.nqubits
            node_mapping_expectation = [i.expectation(final_state) for i in node_mapping]
            gradient = []
            for j in range(len(params)):

                params_left, params_right = copy.deepcopy(params), copy.deepcopy(params)
                params_left[j] = params[j] + (math.pi / 2)
                circuit.set_parameters(params_left)
                final_state_left = circuit()
                node_mapping_expectation_left = [i.expectation(final_state_left) for i in node_mapping]
                params_right[j] = params[j] - (math.pi / 2)
                circuit.set_parameters(params_right)
                final_state_right = circuit()
                node_mapping_expectation_right = [i.expectation(final_state_right) for i in node_mapping]
                derivative = [(node_mapping_expectation_left[l] - node_mapping_expectation_right[l]) / 2 for l in
                              range(len(node_mapping))]
                loss = 0
                for i in adjacency_matrix:
                    loss += adjacency_matrix[i] * self.hyperparameters[0] * qubits * (((np.cosh(
                        node_mapping_expectation[i[0]] * self.hyperparameters[0] * qubits)) ** (-1)) ** 2 *
                                                                                      derivative[i[0]] * np.tanh(
                                node_mapping_expectation[i[1]] * self.hyperparameters[0] * qubits) + ((np.cosh(
                                node_mapping_expectation[i[1]] * self.hyperparameters[0] * qubits)) ** (-1)) ** 2 *
                                                                                      derivative[i[1]] * np.tanh(
                                node_mapping_expectation[i[0]] * self.hyperparameters[0] * qubits))
                penalization = 0
                for i in range(len(node_mapping)):
                    penalization += (node_mapping_expectation[i]) * (derivative[i])
                loss_derivative = loss + self.hyperparameters[1] * 2 * self.max_eigenvalue * (
                        len(node_mapping) / 3 - 2 / 3) * penalization
                gradient.append(loss_derivative)
            return gradient

        def _loss_warmup(params: list, circuit: qibo.models.Circuit, node_mapping: list, solution: list) -> float:
            """
            Function which warm up the circuit enforcing same values of the expectation values of a user-given solution.
            Done using an ad-hoc penalization loss function.
            :param params: parameters used in the circuit
            :param circuit: qibo circuit used for the VQA
            :param node_mapping: encoding of the graph nodes in the observables
            :param solution: best known solution to the problem
            :return: value of the loss (ideally 0)
            """
            # defines loss function with given activation function
            circuit.set_parameters(params)
            final_state = circuit()
            loss = 0
            for i in range(len(node_mapping)):
                loss += (node_mapping[i].expectation(final_state) - 0.1 * solution[i]) ** 2
            return loss

        ################################################################################################################
        ## TENSORFLOW VERSION OF THE FUNCTIONS ABOVE, TO FIX

        def _loss_warmup_tensor(params, circuit, node_mapping, solution):
            circuit.set_parameters(params)
            final_state = circuit.execute().tensor
            nodes = (tf.constant(0), tf.constant(tf.zeros([len(node_mapping)], dtype=np.float64)))
            c = lambda i, p: i < len(node_mapping)
            b = lambda i, p: (i + 1, tf.tensor_scatter_nd_update(p, [[i]], [node_mapping[i].expectation(final_state)]))
            node_mapping_expectation = tf.while_loop(c, b, nodes)[1]
            loss = tf.math.abs(tf.math.subtract(node_mapping_expectation,
                                                tf.math.multiply(tf.constant(0.1, dtype=tf.float64), solution)))
            loss = tf.math.reduce_sum(loss)
            return loss

        def _loss_tensor(params, circuit, tensor_ad_mat_edges, tensor_ad_mat_weights, node_mapping, max_eigenvalue):
            # defines loss function with given activation function
            # Set the parameters of the circuit
            circuit.set_parameters(params)
            # final_state = circuit.execute()

            # Get the final statevector, and its conjugate
            final_state = circuit.execute().tensor
            # final_state_conj = tf.math.conj(final_state)

            # Measure it on the obervables used for the encoding
            # right_side = tf.einsum('ijk,k->ij', node_mapping, final_state)
            # products_vector = tf.einsum('ik,k->i', right_side, final_state_conj)
            # node_mapping_expectation = tf.math.real(products_vector)
            # print(len([i.expectation(final_state) for i in node_mapping]))
            # node_mapping_expectation = tf.constant([i.expectation(final_state).numpy() for i in node_mapping])

            nodes = (tf.constant(0), tf.constant(tf.zeros([len(node_mapping)], dtype=np.float64)))
            c = lambda i, p: i < len(node_mapping)
            b = lambda i, p: (i + 1, tf.tensor_scatter_nd_update(p, [[i]], [node_mapping[i].expectation(final_state)]))
            node_mapping_expectation = tf.while_loop(c, b, nodes)[1]
            first_term = tf.math.tanh(tf.math.multiply(
                tf.math.multiply(tf.constant(1, dtype=tf.float64), tf.constant(circuit.nqubits, dtype=tf.float64)),
                tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 0])))
            second_term = tf.math.tanh(tf.math.multiply(
                tf.math.multiply(tf.constant(1, dtype=tf.float64), tf.constant(circuit.nqubits, dtype=tf.float64)),
                tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 1])))
            loss = tf.math.multiply(tensor_ad_mat_weights, first_term)
            loss = tf.math.multiply(loss, second_term)
            loss = tf.math.reduce_sum(loss)

            penalization_loss = tf.math.multiply(
                tf.math.multiply(tf.constant(68, dtype=tf.float64), tf.math.abs(max_eigenvalue)),
                tf.math.reduce_sum(tf.math.square(node_mapping_expectation)))
            print(tf.math.add(loss, penalization_loss))
            return tf.math.add(loss, penalization_loss)

            # first_term = tf.math.tanh( tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 0]))
            # second_term = tf.math.tanh(tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 1]))
            # loss = tf.math.multiply(tensor_ad_mat_weights, first_term)
            # loss = tf.math.multiply(loss, second_term)
            # loss = tf.math.reduce_sum(loss)
            #
            # return loss
        ################################################################################################################

        def _cut_value(params: list, circuit: qibo.models.Circuit) -> float:
            """
            Function that return the final value of the cut
            :param params: parameters used in the circuit
            :param circuit: qibo circuit used for the VQA
            :return:cut value
            """
            # calculates the cut value (as the name would suggest)
            circuit.set_parameters(params)
            final_state = circuit()
            cut_value = 0
            for i in self.adjacency_matrix:
                cut_value += self.adjacency_matrix[i] * (
                            1 - _round(self.node_mapping[i[0]].expectation(final_state)) * _round(
                        self.node_mapping[i[1]].expectation(final_state))) / 2
            return cut_value

        def _retrieve_solution(params: list, circuit: qibo.models.Circuit) -> list:
            """
            Function which retrieve the unrounded expectation values at the end of the minimization
            :param params:
            :param circuit:
            :return:
            """
            circuit.set_parameters(params)
            final_state = circuit()
            if method == 'sgd':
                first_part = [node.expectation(final_state).numpy for node in self.node_mapping]
            else:
                first_part = [node.expectation(final_state) for node in self.node_mapping]
            return first_part

        def _round(num: float) -> int:
            """
            Rounding function
            :type num: float number
            """
            if num > 0:
                return +1
            elif num < 0:
                return -1
            else:
                raise ValueError('The expectation value of a node is zero.')

        if compile:
            if K.is_custom:
                raise_error(RuntimeError, "Cannot compile VQE that uses custom operators. "
                                          "Set the compile flag to False.")
            for gate in self.circuit.queue:
                _ = gate.cache
            loss = K.compile(_loss)
        else:
            # Do the warmup if required (different depending if tensorflow is used)
            if warmup:
                if K.supports_gradients:
                    tf.debugging.set_log_device_placement(True)
                    loss = _loss_warmup_tensor
                    self.approx_solution = tf.convert_to_tensor(self.approx_solution, dtype=np.float64)
                    node_mapping = self.node_mapping
                    args = (self.circuit, node_mapping, self.approx_solution)
                    options = {'optimizer': 'Adam', "learning_rate": 0.01, "nepochs": 100}
                else:
                    args = (self.circuit, self.node_mapping, self.approx_solution)
                    loss = lambda p, c, ad, af: K.to_numpy(_loss_warmup(p, c, ad, af))

            else:
                # Set up for tensorflow
                if K.supports_gradients:
                    tf.debugging.set_log_device_placement(True)
                    loss = _loss_tensor
                    tensor_ad_mat_egdes = []
                    tensor_ad_mat_weight = []
                    for i in self.adjacency_matrix:
                        tensor_ad_mat_egdes.append([i[0], i[1]])
                        tensor_ad_mat_weight.append(self.adjacency_matrix[i])
                    # node_mapping = [tf.convert_to_tensor(i.matrix) for i in self.node_mapping]
                    # node_mapping = tf.stack(node_mapping)
                    if warmup:
                        self.approx_solution = tf.convert_to_tensor(self.approx_solution)
                    node_mapping = self.node_mapping
                    tensor_ad_mat_edges = tf.convert_to_tensor(tensor_ad_mat_egdes)
                    tensor_ad_mat_weights = tf.convert_to_tensor(tensor_ad_mat_weight, dtype=tf.float64)
                    args = (self.circuit, tensor_ad_mat_edges, tensor_ad_mat_weights, node_mapping,
                            tf.constant(self.max_eigenvalue, dtype=tf.float64))
                    options = {'optimizer': 'Adam', "learning_rate": 0.01, "nepochs": 10000}
                else:
                    loss = _loss
                    # If cma is used, specific set up to get consisten post processing
                    if method == "cma":
                        dtype = getattr(K.np, K._dtypes.get('DTYPE'))
                        loss = lambda p, c, ad, af, nm: dtype(_loss(p, c, ad, af, nm))
                    elif method != "sgd":
                        # Once again, different set up for the different way to get the expectation values
                        if self.expectations_method:
                            from qibo import matrices
                            from functools import reduce
                            from numpy import pi, log10, asarray, kron
                            from qibo import gates as gt
                            previous_string = self.circuit.nqubits * ("I")
                            unitaries = []
                            for current_string in self.node_mapping:
                                qb, matlist = [], []
                                for i, (g1, g2) in enumerate(zip(previous_string, current_string)):
                                    if g1 != g2:
                                        qb.append(i)
                                        matlist.append(getattr(matrices, g2) @ getattr(matrices, g1))
                                if qb:
                                    matrix = reduce(kron, matlist)
                                    unitaries.append(gt.Unitary(matrix, *qb))
                                previous_string = current_string
                            self.node_mapping = unitaries

                        # Actually most used set up for the minimization (every minimizer but keras based or cma)
                        loss = lambda p, c, ad, af, nm: K.to_numpy(_loss(p, c, ad, af, nm))
                        jac = lambda p, c, ad, af, nm: K.to_numpy(_loss_gradient(p, c, ad, af, nm))
                    args = (self.circuit, self.adjacency_matrix, self.activation_function, self.node_mapping)
        # Run the minimization
        result, parameters, extra = self.optimizers.optimize(loss, initial_state,
                                                             args=args,
                                                             method=method, jac=jac, hess=hess, hessp=hessp,
                                                             bounds=bounds, constraints=constraints,
                                                             tol=tol, callback=self._callback, options=options,
                                                             compile=compile,
                                                             processes=processes)

        # Gather last results and return it after the minimization
        solution = _retrieve_solution(parameters, self.circuit)
        cut_value = _cut_value(parameters, self.circuit)
        self.circuit.set_parameters(parameters)
        return result, cut_value, parameters, extra, solution, [_round(i) for i in solution]