from qibo import K, gates
from qibo.config import raise_error

class MultibaseVQE(object):
    """This class implements the variational quantum eigensolver algorithm.

    Args:
        circuit (:class:`qibo.abstractions.circuit.AbstractCircuit`): Circuit that
            implements the variaional ansatz.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian object.

    Example:
        .. testcode::

            import numpy as np
            from qibo import gates, models, hamiltonians
            # create circuit ansatz for two qubits
            circuit = models.Circuit(2)
            circuit.add(gates.RY(0, theta=0))
            # create XXZ Hamiltonian for two qubits
            hamiltonian = hamiltonians.XXZ(2)
            # create VQE model for the circuit and Hamiltonian
            vqe = models.VQE(circuit, hamiltonian)
            # optimize using random initial variational parameters
            initial_parameters = np.random.uniform(0, 2, 1)
            vqe.minimize(initial_parameters)
    """
    from qibo import optimizers

    def __init__(self, circuit):
        """Initialize circuit ansatz and hamiltonian."""
        self.circuit = circuit

    def minimize(self, initial_state, method='Powell', jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None, callback=None,
                 options=None, compile=False, processes=None):
        """Search for parameters which minimizes the hamiltonian expectation.

        Args:
            initial_state (array): a initial guess for the parameters of the
                variational circuit.
            method (str): the desired minimization method.
                See :meth:`qibo.optimizers.optimize` for available optimization
                methods.
            jac (dict): Method for computing the gradient vector for scipy optimizers.
            hess (dict): Method for computing the hessian matrix for scipy optimizers.
            hessp (callable): Hessian of objective function times an arbitrary
                vector for scipy optimizers.
            bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
            constraints (dict): Constraints definition for scipy optimizers.
            tol (float): Tolerance of termination for scipy optimizers.
            callback (callable): Called after each iteration for scipy optimizers.
            options (dict): a dictionary with options for the different optimizers.
            compile (bool): whether the TensorFlow graph should be compiled.
            processes (int): number of processes when using the paralle BFGS method.

        Return:
            The final expectation value.
            The corresponding best parameters.
            The optimization result object. For scipy methods it returns
            the ``OptimizeResult``, for ``'cma'`` the ``CMAEvolutionStrategy.result``,
            and for ``'sgd'`` the options used during the optimization.
        """
        def _measure_one_axis(circuit, axis):
            size= circuit.nqubits
            if axis=='x':
                circuit.add((gates.H(q,register_name="x") for q in range(size)))
            if axis=='y':
                return
            if axis=='z':
                return
            circuit.add((gates.M(q,register_name="y") for q in range(size)))


        def _loss(params, circuit):
            circuit.set_parameters(params)
            _measure_one_axis(circuit, 'z')
            _measure_one_axis(circuit, 'x')
            final_state = circuit(nshots=1)
            return _loss_analitic(final_state)

        def _loss_analitic(final_state):

            return loss

        if compile:
            if K.is_custom:
                raise_error(RuntimeError, "Cannot compile VQE that uses custom operators. "
                                          "Set the compile flag to False.")
            for gate in self.circuit.queue:
                _ = gate.cache
            loss = K.compile(_loss)
        else:
            loss = _loss

        if method == "cma":
            dtype = getattr(K.np, K._dtypes.get('DTYPE'))
            loss = lambda p, c, h: dtype(_loss(p, c, h))
        elif method != "sgd":
            loss = lambda p, c, h: K.to_numpy(_loss(p, c, h))

        result, parameters, extra = self.optimizers.optimize(loss, initial_state,
                                                             args=(self.circuit, self.hamiltonian),
                                                             method=method, jac=jac, hess=hess, hessp=hessp,
                                                             bounds=bounds, constraints=constraints,
                                                             tol=tol, callback=callback, options=options,
                                                             compile=compile, processes=processes)
        self.circuit.set_parameters(parameters)
        return result, parameters, extra