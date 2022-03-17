from qibo import models, gates
from qibo import callbacks

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
    c.add(gates.U3(q, theta=0, phi=0, lam=0,trainable=True) for q in range(size))
    if entanglement == 'basic':
        for l in range(1, p+1):
            c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))
            c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))
            c.add(gates.U3(q, theta=0, phi=0, lam=0,trainable=True) for q in range(size))
    if entanglement == 'interleaved':
        for l in range(1, p+1):
            for k in range(size-1):
                c.add(gates.U3(k))
                c.add(gates.CZ(k, k + 1))
            c.add(gates.U3(size-1, theta=0, phi=0, lam=0))
    if entanglement == 'linear':
        for l in range(1,p+1):
            for k in range(size-1):
                c.add(gates.CZ(k, k+1))
            c.add(gates.U3(q, theta=0, phi=0, lam=0) for q in range(size))

    if entanglement == 'circular':
        for l in range(1,p+1):
            c.add(gates.CZ(size-1, 0))
            for k in range(size-1):
                c.add((gates.CZ(k, k+1)))
            c.add(gates.U3(q, theta=0, phi=0, lam=0) for q in range(size))

    if entanglement == 'circular-interleaved':
        for l in range(1,p+1):
            for k in range(size-1):
                c.add(gates.RZ(k,theta=0, phi=0, lam=0))
                c.add(gates.CZ(k, k + 1))
            c.add(gates.U3(size-1,theta=0, phi=0, lam=0))
            c.add(gates.CZ(size - 1, 0))

    return c

def var_form_RY(size = 6, p: int = 0, entanglement='basic'):
    c = models.Circuit(size)
    if entanglement == 'basic':
        for l in range(0, p+1):
            c.add(gates.RY(q, theta=0,trainable=True) for q in range(size))
            c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))
            c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))

    return c

def overlap_retriver(ansaz, true_vector ):
    overlap = callbacks.Overlap(true_vector)
    ansaz.add(gates.CallbackGate(overlap))
    return ansaz