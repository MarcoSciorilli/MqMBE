from numpy.random import uniform, randint
from numpy import ceil, floor, ndindex, tanh
from qibo.symbols import I, X, Y, Z
from qibo.config import raise_error
from qibo import K
from itertools import combinations
import tensorflow as tf
import numpy as np

class MultibaseVQA(object):
    from qibo import optimizers
    from qibo import K
    def __init__(self, circuit, adjacency_matrix):
        self.activation_function = self.linear_activation
        self.circuit = circuit
        self.adjacency_matrix = adjacency_matrix

    @staticmethod
    def get_num_qubits(num_nodes, pauli_string_length, ratio_total_words):
        # return the number of qubits necessary
        return int(ceil(num_nodes / round((4 ** pauli_string_length - 1) * ratio_total_words)) * pauli_string_length)

    @staticmethod
    def linear_activation(x):
        return x

    @staticmethod
    def my_activation(x):
        if x > 0:
            return (tanh((x-(1/np.sqrt(45)))*45)+1)/2
        else:
            return -(tanh(-(x+(1/np.sqrt(45)))*45)+1)/2

    def encode_nodes(self, num_nodes, pauli_string_length, ratio_total_words, compression=None, lower_order_terms=None):

        def get_pauli_word(indices, k):
            # Generate pauli string corresponding to indices
            # where (0, 1, 2, 3) -> 1XYZ and so on
            from qibo import hamiltonians
            import numpy as np
            pauli_matrices = [I, X, Y, Z]
            word = 1
            for qubit, i in enumerate(indices):
                word *= pauli_matrices[i](qubit + int(k))
            return hamiltonians.SymbolicHamiltonian(word)

        if compression is None:
            # count number of strings per word length to be used
            num_strings = round((4 ** pauli_string_length - 1) * ratio_total_words)
            # generate list of all indices of given length
            indices = list(ndindex(*[4] * pauli_string_length))
            pauli_strings = [x for _, x in sorted(zip([len(i) - i.count(0) for i in indices], indices))]

            self.node_mapping = [
                get_pauli_word(pauli_strings[int(i % num_strings + 1)], pauli_string_length * floor(i / num_strings))
                for i
                in range(num_nodes)]
        else:
            pauli_strings = self._pauli_string(pauli_string_length, compression)
            num_strings = len(pauli_strings)
            # position i stores string corresponding to the i-th node.
            if lower_order_terms is None:
                self.node_mapping = [
                    get_pauli_word(pauli_strings[int(i % num_strings)], pauli_string_length * floor(i / num_strings)) for i
                    in range(pauli_string_length*3, num_nodes + pauli_string_length*3)]
            else:
                self.node_mapping = [
                    get_pauli_word(pauli_strings[int(i % num_strings)], pauli_string_length * floor(i / num_strings)) for i
                    in range(num_nodes)]
        return ceil(num_nodes / num_strings)

    def set_activation(self, function):
        self.activation_function = function

    def set_circuit(self, circuit):
        self.circuit = circuit

    @staticmethod
    def _pauli_string(pauli_string_length, compression):
        pauli_string = []
        for i in range(1, 4):
            for k in range(1, compression + 1):
                comb = combinations(list(range(pauli_string_length)), k)
                for positions in comb:
                    instance = [0] * pauli_string_length
                    for index in positions:
                        instance[index] = i
                    pauli_string.append(tuple(instance))

        return sorted(pauli_string, key=lambda tup: tup.count(0), reverse=True)

    def minimize(self, initial_state, method='Powell', jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None, callback=None,
                 options=None, processes=None, compile=False):

        def _loss(params, circuit, adjacency_matrix, activation_function, node_mapping):
            # defines loss function with given activation function
            circuit.set_parameters(params)
            final_state = circuit()
            loss = 0
            for i in adjacency_matrix:
                loss += adjacency_matrix[i] * activation_function(node_mapping[i[0]].expectation(final_state, normalize=True)) \
                        * activation_function(node_mapping[i[1]].expectation(final_state, normalize=True))
            return loss



        def _loss_tensor(params, circuit, tensor_ad_mat_edges, tensor_ad_mat_weights, node_mapping):
            # defines loss function with given activation function
            circuit.set_parameters(params)
            final_state = circuit.execute().tensor
            final_state_conj = tf.math.conj(final_state)
            right_side = tf.einsum('ijk,k->ij', node_mapping, final_state)
            products_vector = tf.einsum('ik,k->i', right_side, final_state_conj)
            node_mapping_expectation = tf.math.real(products_vector)

            first_term = tf.math.tanh(tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 0]))
            second_term = tf.math.tanh(tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 1]))
            loss = tf.math.multiply(tensor_ad_mat_weights, first_term)
            loss = tf.math.multiply(loss, second_term)
            loss = tf.math.reduce_sum(loss)
            return loss

        def _cut_value(params, circuit):
            # calculates the cut value (as the name would suggest)
            circuit.set_parameters(params)
            final_state = circuit()
            cut_value = 0
            for i in self.adjacency_matrix:
                cut_value += self.adjacency_matrix[i] * (1 \
                                                         - _round(self.node_mapping[i[0]].expectation(final_state, normalize=True)) \
                                                         * _round(self.node_mapping[i[1]].expectation(final_state, normalize=True))) / 2
            return cut_value

        def _retrive_solution(params, circuit):
            circuit.set_parameters(params)
            final_state = circuit()
            first_part = [node.expectation(final_state, normalize=True) for node in self.node_mapping]
            return first_part

        def _round(num):
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
            if K.supports_gradients:
                tf.debugging.set_log_device_placement(True)
                loss = _loss_tensor
                tensor_ad_mat_egdes = []
                tensor_ad_mat_weight = []
                for i in self.adjacency_matrix:
                    tensor_ad_mat_egdes.append([i[0], i[1]])
                    tensor_ad_mat_weight.append(self.adjacency_matrix[i])
                node_mapping = [tf.convert_to_tensor(i.matrix) for i in self.node_mapping]
                node_mapping = tf.stack(node_mapping)
                tensor_ad_mat_edges = tf.convert_to_tensor(tensor_ad_mat_egdes)
                tensor_ad_mat_weights = tf.convert_to_tensor(tensor_ad_mat_weight)
                args = (self.circuit, tensor_ad_mat_edges, tensor_ad_mat_weights, node_mapping)
                # TO HAVE ACCEPTABE COMPUTATIONAL TIME, ACCURACY BASED STOPPING CRITERION MUST BE IMPLEMENTED
                options = { 'optimizer' : 'SGD'}
            else:
                loss = _loss
                args = (self.circuit, self.adjacency_matrix, self.activation_function,self.node_mapping)

        if method == "cma":
            dtype = getattr(K.np, K._dtypes.get('DTYPE'))
            loss = lambda p, c, ad, af, nm: dtype(_loss(p, c, ad, af, nm))
        elif method != "sgd":
            loss = lambda p, c, ad, af, nm: K.to_numpy(_loss(p, c, ad, af, nm))

        result, parameters, extra = self.optimizers.optimize(loss, initial_state,
                                                             args=args,
                                                             method=method, jac=jac, hess=hess, hessp=hessp,
                                                             bounds=bounds, constraints=constraints,
                                                             tol=tol, callback=callback, options=options,
                                                             compile=compile,
                                                             processes=processes)
        solution = _retrive_solution(parameters, self.circuit)
        cut_value = _cut_value(parameters, self.circuit)
        self.circuit.set_parameters(parameters)
        return result, cut_value, parameters, extra, solution, [_round(i) for i in solution]
