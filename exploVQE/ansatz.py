from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_optimization import QuadraticProgram
from math import pi
from qibo import models, gates
from qibo import callbacks


def qaoa_circuit(qubo: QuadraticProgram, p: int = 1):
    """
    Given a QUBO instance and the number of layers p, constructs the corresponding parameterized QAOA circuit with p layers.
    Args:
        qubo: The quadratic program instance
        p: The number of layers in the QAOA circuit
    Returns:
        The parameterized QAOA circuit
    """
    size = len(qubo.variables)
    qubo_matrix = qubo.objective.quadratic.to_array(symmetric=True)
    qubo_linearity = qubo.objective.linear.to_array()

    # Prepare the quantum and classical registers
    qaoa_circuit = QuantumCircuit(size, size)
    # Apply the initial layer of Hadamard gates to all qubits
    qaoa_circuit.h(range(size))

    # Create the parameters to be used in the circuit
    gammas = ParameterVector('gamma', p)
    betas = ParameterVector('beta', p)
    # Outer loop to create each layer
    for i in range(p):
        for k in range(size):
            qubo_matrix_sum = 0
            for q in range(size):
                qubo_matrix_sum += qubo_matrix[k, q]

            # Apply R_Z rotational gates from cost layer
            qaoa_circuit.rz((qubo_linearity[k] + qubo_matrix_sum) * gammas[i], k)

        # Apply R_ZZ rotational gates for entangled qubit rotations from cost layer
        for l in range(size):
            for m in range(size):
                if l != m:
                    qaoa_circuit.rzz(0.5 * qubo_matrix[l, m] * gammas[i], l, m)

        # Apply single qubit X - rotations with angle 2*beta_i to all qubits
        for f in range(size):
            qaoa_circuit.rx(2 * betas[i], f)
    return qaoa_circuit


def circuit_none(size=6):
    """
    Given a QUBO instance and the number of layers p, costruct a circuit made of only a layer of Hadamard gates.
    Args:
        qubo: The quadratic program instance
    Returns:
        The circuit
    """
    c = models.Circuit(size)
    c.add((gates.H(q) for q in range(size)))
    return c

def var_form(size = 6, p: int = 0, entanglement='basic'):
    c = models.Circuit(size)
    c.add((gates.RZ(q, theta=0) for q in range(size)))
    c.add((gates.RX(q, theta=-pi / 2, trainable=False) for q in range(size)))
    c.add((gates.RZ(q, theta=0) for q in range(size)))
    c.add((gates.RX(q, theta=pi / 2, trainable=False) for q in range(size)))
    c.add((gates.RZ(q, theta=0) for q in range(size)))
    if entanglement == 'basic':
        for l in range(1, p):
            c.add((gates.CZ(q, q + 1) for q in range(0, size - 1, 2)))
            c.add((gates.CZ(q, q + 1) for q in range(1, size - 2, 2)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))
            c.add((gates.RX(q, theta=-pi/2, trainable=False) for q in range(size)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))
            c.add((gates.RX(q, theta=pi/2, trainable=False) for q in range(size)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))
    if entanglement == 'interleaved':
        for l in range(1, p):
            for k in range(size-1):
                c.add((gates.RZ(k, theta=0)))
                c.add((gates.RX(k, theta=-pi/2, trainable=False)))
                c.add((gates.RZ(k, theta=0)))
                c.add((gates.RX(k, theta=pi/2, trainable=False)))
                c.add((gates.RZ(k, theta=0)))
                c.add((gates.CZ(k, k + 1)))
            c.add((gates.RZ(size-1, theta=0)))
            c.add((gates.RX(size-1, theta=-pi / 2, trainable=False)))
            c.add((gates.RZ(size-1, theta=0)))
            c.add((gates.RX(size-1, theta=pi / 2, trainable=False)))
            c.add((gates.RZ(size-1, theta=0)))
    if entanglement == 'linear':
        for l in range(1,p):
            for k in range(size-1):
                c.add((gates.CZ(k, k+1)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))
            c.add((gates.RX(q, theta=-pi/2, trainable=False) for q in range(size)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))
            c.add((gates.RX(q, theta=pi/2, trainable=False) for q in range(size)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))

    if entanglement == 'circular':
        for l in range(1,p):
            c.add((gates.CZ(size-1, 0)))
            for k in range(size-1):
                c.add((gates.CZ(k, k+1)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))
            c.add((gates.RX(q, theta=-pi/2, trainable=False) for q in range(size)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))
            c.add((gates.RX(q, theta=pi/2, trainable=False) for q in range(size)))
            c.add((gates.RZ(q, theta=0) for q in range(size)))
    if entanglement == 'circular-interleaved':
        for l in range(1,p):
            for k in range(size-1):
                c.add((gates.RZ(k, theta=0)))
                c.add((gates.RX(k, theta=-pi/2, trainable=False)))
                c.add((gates.RZ(k, theta=0)))
                c.add((gates.RX(k, theta=pi/2, trainable=False)))
                c.add((gates.RZ(k, theta=0)))
                c.add((gates.CZ(k, k + 1)))
            c.add((gates.RZ(size-1, theta=0)))
            c.add((gates.RX(size-1, theta=-pi / 2, trainable=False)))
            c.add((gates.RZ(size-1, theta=0)))
            c.add((gates.RX(size-1, theta=pi / 2, trainable=False)))
            c.add((gates.RZ(size-1, theta=0)))
            c.add((gates.CZ(size - 1, 0)))

    return c

def overlap_retriver(ansaz, true_vector ):
    overlap = callbacks.Overlap(true_vector)
    ansaz.add(gates.CallbackGate(overlap))
    return ansaz