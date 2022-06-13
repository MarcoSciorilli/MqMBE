import math
import copy
from numpy.random import uniform, randint
from numpy import ceil, floor, ndindex, tanh
from qibo.symbols import I, X, Y, Z
from qibo.config import raise_error
from qibo import K
from itertools import combinations
import tensorflow as tf
import numpy as np
import random
from itertools import combinations, product
import mpmath
global expectations_method
expectations_method = False

class MultibaseVQA(object):
    from qibo import optimizers
    from qibo import K
    def __init__(self, circuit, adjacency_matrix, max_eigenvalue, hyperparameters):
        self.activation_function = self.linear_activation
        self.circuit = circuit
        self.adjacency_matrix = adjacency_matrix
        self.parameter_iteration = []
        self.max_eigenvalue = max_eigenvalue
        self.hyperparameters = hyperparameters
        self.approx_solution = None
        self.ratio = 0
        self.node_expectation_mapping = 0

    @staticmethod
    def get_num_qubits(num_nodes, pauli_string_length, ratio_total_words):
        # return the number of qubits necessary
        return int(ceil(num_nodes / round((4 ** pauli_string_length - 1) * ratio_total_words)) * pauli_string_length)

    @staticmethod
    def linear_activation(x):
        return x

    @staticmethod
    def picewise_sin(x):
        if np.pi*x/2 > 1:
            return 1
        if np.pi*x/2 < -1:
            return -1
        else:
            return np.sin(np.pi*x/2)


    @staticmethod
    def my_activation(x):
        if x > 0:
            return (tanh((x-(0.5))*(10))+1)/2
        else:
            return -(tanh(-(x+(0.5))*(10))+1)/2

    def _callback(self, x):
        self.parameter_iteration.append(x)

    def encode_nodes(self, num_nodes, pauli_string_length, ratio_total_words=None, compression=None, lower_order_terms=False, shuffle=True, seed=0, same_letter=True):

        def get_pauli_word(indices, k, qubits=None):
            # Generate pauli string corresponding to indices
            # where (0, 1, 2, 3) -> 1XYZ and so on
            from qibo import hamiltonians
            import numpy as np



            if expectations_method:
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
                pauli_matrices = np.array([I, X, Y, Z])
                word = np.int(1)
                for qubit, i in enumerate(indices):
                    word *= pauli_matrices[i](qubit + int(k))
                if qubits:
                    qubits_list = list(range(qubits))
                    qubits_list.remove(int(k))
                    for j in qubits_list:
                        word *= pauli_matrices[0](j)
                return hamiltonians.SymbolicHamiltonian(word)

        if compression is None:

            pauli_strings = self._pauli_string_same_letter(pauli_string_length, 1, True, shuffle, seed, pauli_letters=int(ratio_total_words*3+1))

            num_strings = len(pauli_strings)

            # position i stores string corresponding to the i-th node.
            self.node_mapping = [
                get_pauli_word(pauli_strings[int(i % num_strings)], pauli_string_length * floor(i / num_strings), self.get_num_qubits(num_nodes, pauli_string_length,ratio_total_words)) for i
                in range(num_nodes)]
        else:
            if same_letter:
                pauli_strings = self._pauli_string_same_letter(pauli_string_length, compression,lower_order_terms, shuffle, seed)
            else:
                pauli_strings = self._random_pauli_string(pauli_string_length, compression, lower_order_terms,
                                                               shuffle, seed)

            num_strings = len(pauli_strings)
            # position i stores string corresponding to the i-th node.
            if expectations_method:
                node_mapping= []
                for current_string in product(["I", "X", "Y", "Z"], repeat=9):
                    current_string_dic = {}
                    for j in current_string:
                        if j in current_string_dic:
                            current_string_dic[j] = current_string_dic[j] + 1
                        else:
                            current_string_dic[j] = 0
                    if len(current_string_dic) > 2 or 'I' not in current_string_dic or current_string_dic[
                        'I'] < 9 - 3 or current_string_dic['I'] > 9 - 3:
                        continue
                    else:
                        node_mapping.append(current_string)
                import scipy
                for i in range(len(node_mapping) - 1):
                    smallest = 1000
                    for j in range(len(node_mapping) - 1, i - 1, -1):
                        if scipy.spatial.distance.hamming(node_mapping[i], node_mapping[j]) < smallest:
                            smallest = j
                    node_mapping[i + 1], node_mapping[smallest] = node_mapping[smallest], node_mapping[i + 1]
                self.node_mapping = node_mapping
            else:
                self.node_mapping = [
                    get_pauli_word(pauli_strings[int(i % num_strings)], pauli_string_length * floor(i / num_strings))
                    for i
                    in range(num_nodes)]
        return ceil(num_nodes / num_strings)

    def set_activation(self, function):
        self.activation_function = function

    def set_circuit(self, circuit):
        self.circuit = circuit

    def set_approx_solution(self, solution):
        self.approx_solution = solution

    @staticmethod
    def _pauli_string_same_letter(pauli_string_length, order, lower_order_terms, shuffle, seed, pauli_letters=4):
        pauli_string = []
        if lower_order_terms:
            smallest_lenght = 1
        else:
            smallest_lenght = order
        for i in range(1, pauli_letters):
            for k in range(smallest_lenght, order + 1):
                comb = combinations(list(range(pauli_string_length)), k)
                for positions in comb:
                    instance = [0] * pauli_string_length
                    for index in positions:
                        instance[index] = i
                    pauli_string.append(tuple(instance))
        if shuffle:
            random.seed(seed)
            random.shuffle(pauli_string)
        #return sorted(pauli_string, key=lambda tup: tup.count(0), reverse=True)
        return pauli_string

    @staticmethod
    def _random_pauli_string(pauli_string_length, order, lower_order_terms, shuffle, seed):
        pauli_tuples = [(i,j) for i in range(1,4) for j in range(pauli_string_length)]
        if lower_order_terms:
            smallest_lenght = 1
        else:
            smallest_lenght = order
        total_combinations = []
        for i in range(smallest_lenght, order + 1):
            total_combinations = total_combinations +(list(combinations(pauli_tuples, i)))
        pauli_string = []
        for comb in total_combinations:
            instance = [0] * pauli_string_length
            for j in comb:
                instance[j[1]] = j[0]
                pauli_string.append(instance)
        if shuffle:
            random.seed(seed)
            random.shuffle(pauli_string)
        return pauli_string
    def minimize(self, initial_state, method='Powell', jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None, callback=None,
                 options=None, processes=None, compile=False, warmup = False):

        def _loss(params, circuit, adjacency_matrix, activation_function, node_mapping):
            # defines loss function with given activation function
            circuit.set_parameters(params)
            qubits = circuit.nqubits
            loss = 0
            if expectations_method:
                final_state = circuit().numpy()
                tstate = np.copy(final_state)
                representation = []
                for gate in node_mapping:
                    tstate = gate(tstate)
                    representation.append(np.conj(tstate).dot(final_state).real)
                pauli_basis_representation = asarray(representation)
                node_mapping_expectation = pauli_basis_representation
            else:
                final_state = circuit()
                node_mapping_expectation = [i.expectation(final_state) for i in node_mapping]
            self.node_expectation_mapping = node_mapping_expectation
            for i in adjacency_matrix:
                loss += adjacency_matrix[i] * activation_function(node_mapping_expectation[i[0]]*self.hyperparameters[0]*qubits) \
                        * activation_function(node_mapping_expectation[i[1]]*self.hyperparameters[0]*qubits)

            penalization = 0
            for i in range(len(node_mapping)):
                penalization += ((node_mapping_expectation[i])**2)
            loss_1 = 0
            for i in adjacency_matrix:
                loss_1 += abs(adjacency_matrix[i] * activation_function(node_mapping_expectation[i[0]]*self.hyperparameters[0]*qubits) \
                        * activation_function(node_mapping_expectation[i[1]]*self.hyperparameters[0]*qubits))
            self.ratio = loss_1/(self.hyperparameters[1] * abs(self.max_eigenvalue)*penalization)
            # print('LOSS: ',loss + self.hyperparameters[1] *(len(node_mapping)/3-0.6666666)* abs(self.max_eigenvalue)*penalization)
            return loss + self.hyperparameters[1] *(len(node_mapping)/3-0.6666666)* abs(self.max_eigenvalue)*penalization


        def _loss_derivative(params, circuit, adjacency_matrix, activation_function, node_mapping):
            # defines loss function with given activation function
            circuit.set_parameters(params)
            final_state = circuit()
            qubits = circuit.nqubits

            node_mapping_expectation = [i.expectation(final_state) for i in node_mapping]
            gradient = []
            for j in range(len(params)):

                params_left, params_right = copy.deepcopy(params), copy.deepcopy(params)
                params_left[j] = params[j] + (math.pi/2)
                circuit.set_parameters(params_left)
                final_state_left = circuit()
                node_mapping_expectation_left = [i.expectation(final_state_left) for i in node_mapping]
                params_right[j] = params[j] - (math.pi/2)
                circuit.set_parameters(params_right)
                final_state_right = circuit()
                node_mapping_expectation_right = [i.expectation(final_state_right) for i in node_mapping]
                derivative = [(node_mapping_expectation_left[l]-node_mapping_expectation_right[l])/2 for l in range(len(node_mapping))]
                loss = 0
                for i in adjacency_matrix:
                    loss += adjacency_matrix[i] * self.hyperparameters[0] * qubits * (((np.cosh(node_mapping_expectation[i[0]] * self.hyperparameters[0]*qubits))**(-1)) ** (2) * derivative[i[0]] \
                            * np.tanh(node_mapping_expectation[i[1]] * self.hyperparameters[0] * qubits) + ((np.cosh(node_mapping_expectation[i[1]] * self.hyperparameters[0] * qubits))**(-1)) ** (2) * derivative[i[1]] \
                            * np.tanh(node_mapping_expectation[i[0]] * self.hyperparameters[0] * qubits))

                penalization = 0
                for i in range(len(node_mapping)):
                    penalization += (node_mapping_expectation[i])*(derivative[i])
                loss_derivative = loss + self.hyperparameters[1] * 2 * (len(node_mapping) / 3 - 0.6666666) * abs(self.max_eigenvalue) * penalization
                gradient.append(loss_derivative)
            return gradient

        def _loss_warmup(params, circuit, node_mapping, solution):
            # defines loss function with given activation function
            circuit.set_parameters(params)
            final_state = circuit()
            loss = 0
            for i in range(len(node_mapping)):
                loss += (node_mapping[i].expectation(final_state) - 0.1*solution[i])**2
            return loss

        def _loss_warmup_tensor(params, circuit, node_mapping, solution):
            circuit.set_parameters(params)
            final_state = circuit.execute().tensor
            nodes = (tf.constant(0), tf.constant(tf.zeros([len(node_mapping)], dtype=np.float64)))
            c = lambda i, p: i < len(node_mapping)
            b = lambda i, p: (i + 1, tf.tensor_scatter_nd_update(p, [[i]], [node_mapping[i].expectation(final_state)]))
            node_mapping_expectation = tf.while_loop(c, b, nodes)[1]
            loss = tf.math.abs(tf.math.subtract(node_mapping_expectation,tf.math.multiply(tf.constant(0.1, dtype=tf.float64), solution)))
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
            first_term = tf.math.tanh(tf.math.multiply(tf.math.multiply(tf.constant(1, dtype=tf.float64),tf.constant(circuit.nqubits, dtype=tf.float64)), tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 0])))
            second_term = tf.math.tanh(tf.math.multiply(tf.math.multiply(tf.constant(1, dtype=tf.float64),tf.constant(circuit.nqubits, dtype=tf.float64)),tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 1])))
            loss = tf.math.multiply(tensor_ad_mat_weights, first_term)
            loss = tf.math.multiply(loss, second_term)
            loss = tf.math.reduce_sum(loss)

            penalization_loss = tf.math.multiply(tf.math.multiply(tf.constant(68,dtype=tf.float64), tf.math.abs(max_eigenvalue)), tf.math.reduce_sum(tf.math.square(node_mapping_expectation)))
            print(tf.math.add(loss, penalization_loss))
            return tf.math.add(loss, penalization_loss)

            # first_term = tf.math.tanh( tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 0]))
            # second_term = tf.math.tanh(tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 1]))
            # loss = tf.math.multiply(tensor_ad_mat_weights, first_term)
            # loss = tf.math.multiply(loss, second_term)
            # loss = tf.math.reduce_sum(loss)
            #
            # return loss

        def _cut_value(params, circuit):
            # calculates the cut value (as the name would suggest)
            circuit.set_parameters(params)
            final_state = circuit()
            cut_value = 0
            for i in self.adjacency_matrix:
                cut_value += self.adjacency_matrix[i] * (1 \
                                                         - _round(self.node_mapping[i[0]].expectation(final_state)) \
                                                         * _round(self.node_mapping[i[1]].expectation(final_state))) / 2
            return cut_value

        def _retrive_solution(params, circuit):
            circuit.set_parameters(params)
            final_state = circuit()
            if method == 'sgd':
                first_part = [node.expectation(final_state).numpy for node in self.node_mapping]
            else:
                first_part = [node.expectation(final_state) for node in self.node_mapping]
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
            if warmup:
                if K.supports_gradients:
                    tf.debugging.set_log_device_placement(True)
                    loss = _loss_warmup_tensor
                    self.approx_solution = tf.convert_to_tensor(self.approx_solution,  dtype=np.float64)
                    node_mapping = self.node_mapping
                    args = (self.circuit,  node_mapping,self.approx_solution)
                    options = {'optimizer': 'Nadam', "learning_rate": 0.01, "nepochs": 100}
                else:
                    args = (self.circuit, self.node_mapping, self.approx_solution)
                    loss = lambda p, c, ad, af,: K.to_numpy(_loss_warmup(p, c, ad, af))


            else:
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
                    args = (self.circuit, tensor_ad_mat_edges, tensor_ad_mat_weights, node_mapping, tf.constant(self.max_eigenvalue, dtype=tf.float64))
                    options = { 'optimizer' : 'Nadam', "learning_rate": 0.01,  "nepochs": 10000}
                else:
                    loss = _loss
                    if method == "cma":
                        dtype = getattr(K.np, K._dtypes.get('DTYPE'))
                        loss = lambda p, c, ad, af, nm: dtype(_loss(p, c, ad, af, nm))
                    elif method != "sgd":
                        if expectations_method:
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
                        loss = lambda p, c, ad, af, nm: K.to_numpy(_loss(p, c, ad, af, nm))
                        jac = lambda p, c, ad, af, nm: K.to_numpy(_loss_derivative(p, c, ad, af, nm))
                    args = (self.circuit, self.adjacency_matrix, self.activation_function, self.node_mapping)


        result, parameters, extra = self.optimizers.optimize(loss, initial_state,
                                                             args=args,
                                                             method=method, jac=jac, hess=hess, hessp=hessp,
                                                             bounds=bounds, constraints=constraints,
                                                             tol=tol, callback=self._callback, options=options,
                                                             compile=compile,
                                                             processes=processes)
        solution = _retrive_solution(parameters, self.circuit)
        cut_value = _cut_value(parameters, self.circuit)
        self.circuit.set_parameters(parameters)
        print(cut_value)
        return result, cut_value, parameters, extra, solution, [_round(i) for i in solution]
