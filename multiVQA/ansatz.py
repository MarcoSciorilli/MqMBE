from qibo import models, gates
from qibo import callbacks


def circuit_none(size: int = 6) -> models.Circuit:
    """
    Given a number of qubits, construct a circuit made of only a layer of Hadamard gates (for baseline).
    :param size: number of qubits
    :return: A qibo circuit
    """
    c = models.Circuit(size)
    c.add((gates.H(q) for q in range(size)))
    return c


def var_form(size=6, p: int = 0, entanglement='basic'):
    c = models.Circuit(size)
    c.add(gates.U3(q, theta=0, phi=0, lam=0, trainable=True) for q in range(size))
    if entanglement == 'circular':
        for l in range(1, p + 1):
            c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))
            if size % 2:
                c.add(gates.CZ(q, q + 1) for q in range(1, size, 2))
                c.add(gates.CZ(size - 1, 0))
            else:
                c.add(gates.CZ(size - 1, 0))
                c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))
            c.add(gates.U3(q, theta=0, phi=0, lam=0, trainable=True) for q in range(size))

    if entanglement == 'basic':
        for l in range(1, p + 1):
            if size == 2:
                c.add(gates.CZ(0, 1))
            elif size % 2:
                c.add(gates.CZ(q, q + 1) for q in range(0, size - 2, 2))
                c.add(gates.CZ(q, q + 1) for q in range(1, size - 1, 2))
            else:
                c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))
                c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))
            c.add(gates.U3(q, theta=0, phi=0, lam=0, trainable=True) for q in range(size))

    if entanglement == 'article_circular':
        for l in range(1, p + 1):
            if size == 2:
                c.add(gates.CZ(0, 1))
            elif l%2:
                if size%2:
                    c.add(gates.CZ(q, q + 1) for q in range(0, size - 2, 2))
                else:
                    c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))

            else:
                if size % 2:
                    c.add(gates.CZ(q, q + 1) for q in range(1, size-1, 2))
                else:
                    c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))
            c.add(gates.CZ(size - 1, 0))
            c.add(gates.RY(q, theta=0, trainable=True) for q in range(size))

    if entanglement == 'article':
        for l in range(1, p + 1):
            if size == 2:
                c.add(gates.CZ(0, 1))
            elif l%2:
                if size%2:
                    c.add(gates.CZ(q, q + 1) for q in range(0, size - 2, 2))
                else:
                    c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))

            else:
                if size % 2:
                    c.add(gates.CZ(q, q + 1) for q in range(1, size-1, 2))
                else:
                    c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))
            c.add(gates.RY(q, theta=0, trainable=True) for q in range(size))

    return c


def overlap_retriver(ansaz, true_vector):
    overlap = callbacks.Overlap(true_vector)
    ansaz.add(gates.CallbackGate(overlap))
    return ansaz
