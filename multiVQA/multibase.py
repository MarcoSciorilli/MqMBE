from numpy.random import uniform, randint
from numpy import ceil, floor, ndindex
from qibo.symbols import X, Y, Z
import networkx as nx

class MultibaseVQA(object):
    from qibo import optimizers

    def __init__(self, circuit, adjacency_matrix):
        self.activation_function = None
        self.circuit = circuit
        self.adjacency_matrix = adjacency_matrix

    @staticmethod
    def get_num_qubits(num_nodes, pauli_string_length, ratio_total_words):
        # return the number of qubits necessary
        return int(ceil(num_nodes / round((4 ** pauli_string_length - 1) * ratio_total_words)) * pauli_string_length)

    @staticmethod
    def linear_activation(x):
        return x

    def encode_nodes(self, num_nodes, pauli_string_length, ratio_total_words):

        def get_pauli_word(indices, k):
            # Generate pauli string corresponding to indices
            # where (0, 1, 2, 3) -> 1XYZ and so on
            from qibo import hamiltonians
            pauli_matrices = [1, X, Y, Z]
            word = 1
            for qubit, i in enumerate(indices):
                if pauli_matrices[i] == 1:
                    continue
                word *= pauli_matrices[i](qubit + int(k))
            return hamiltonians.SymbolicHamiltonian(word)

        # count number of strings per word length to be used
        num_strings = round((4 ** pauli_string_length - 1) * ratio_total_words)
        # generate list of all indices of given length
        indices = list(ndindex(*[4] * pauli_string_length))
        pauli_strings = [x for _, x in sorted(zip([len(i) - i.count(0) for i in indices], indices))]
        # position i stores string corresponding to the i-th node.
        self.node_mapping = [
            get_pauli_word(pauli_strings[int(i % num_strings) + 1], pauli_string_length * floor(i / num_strings)) for i
            in range(num_nodes)]
        return ceil(num_nodes / num_strings)

    def set_activation(self, function):
        self.activation_function = function

    def minimize(self, initial_state, method='Powell', jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None, callback=None,
                 options=None, processes=None):

        def _loss(params, circuit, activation_function):
            # defines loss function with given activation function
            circuit.set_parameters(params)
            final_state = circuit()
            loss = 0
            for i in self.adjacency_matrix:
                loss += self.adjacency_matrix[i] * activation_function(self.node_mapping[i[0]].expectation(final_state)) \
                         * activation_function(self.node_mapping[i[1]].expectation(final_state))

            return loss

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
            first_part = [node.expectation(final_state) for node in self.node_mapping]
            return first_part

        def _round(num):
            if num > 0:
                return +1
            elif num < 0:
                return -1
            else:
                raise ValueError('The expectation value of a node is zero.')

        result, parameters, extra = self.optimizers.optimize(_loss, initial_state,
                                                             args=(self.circuit, self.activation_function),
                                                             method=method, jac=jac, hess=hess, hessp=hessp,
                                                             bounds=bounds, constraints=constraints,
                                                             tol=tol, callback=callback, options=options,
                                                             processes=processes)
        solution = _retrive_solution(parameters, self.circuit)
        cut_value = _cut_value(parameters, self.circuit)
        self.circuit.set_parameters(parameters)
        return result, cut_value, parameters, extra, solution, [_round(i) for i in solution]