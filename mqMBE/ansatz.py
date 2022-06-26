from qibo import models, gates
from qibo import callbacks
import math

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
    if entanglement == 'simple' or entanglement == 'double_entanglement' or entanglement == 'rotating' or entanglement == '2D' :
        c.add(gates.RY(q, theta=0, trainable=True) for q in range(size))
    elif entanglement == 'rotating_U3':
        pass
    else:
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

    if entanglement == 'article' or entanglement == 'simple':
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

    if entanglement == 'double_entanglement':
        for l in range(1, p + 1):
            if size == 2:
                c.add(gates.CZ(0, 1))
            else:
                if size % 2:
                    c.add(gates.CZ(q, q + 1) for q in range(0, size - 2, 2))
                    c.add(gates.CZ(q, q + 1) for q in range(1, size-1, 2))
                else:
                    c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))
                    c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))
            c.add(gates.RY(q, theta=0, trainable=True) for q in range(size))

    if entanglement == 'rotating':
        for l in range(1, p + 1):
            if size == 2:
                c.add(gates.CZ(0, 1))
            else:
                entang_list=[ q for q in range(l-1, size + l-1)]
                def refit(entang_list, size):
                    for i in range(len(entang_list)):
                        if entang_list[i] > size - 1:
                            entang_list[i] = entang_list[i] - size
                    if all(q < size for q in entang_list):
                        return entang_list
                    else:
                        return refit(entang_list, size)
                entang_list = refit(entang_list, size)
                c.add(gates.CZ(entang_list[q-1], entang_list[q]) for q in range(1, len(entang_list), 2))
            c.add(gates.RY(q, theta=0, trainable=True) for q in range(size))

    if entanglement == 'rotating_U3':
        for l in range(1, p + 1):
            if size == 2:
                c.add(gates.CZ(0, 1))
            else:
                entang_list=[ q for q in range(l-1, size + l-1)]
                def refit(entang_list, size):
                    for i in range(len(entang_list)):
                        if entang_list[i] > size - 1:
                            entang_list[i] = entang_list[i] - size
                    if all(q < size for q in entang_list):
                        return entang_list
                    else:
                        return refit(entang_list, size)
                entang_list = refit(entang_list, size)
                c.add(gates.CZ(entang_list[q-1], entang_list[q]) for q in range(1, len(entang_list), 2))
            c.add(gates.U3(q, theta=0, phi=0, lam=0, trainable=True) for q in range(size))

    if entanglement == '2D':
        for l in range(1, p + 1):
            if size == 2:
                c.add(gates.CZ(0, 1))
            else:
                layer = l
                qubits = size
                size = math.ceil(math.sqrt(size))
                edge_list = []
                if size % 2:
                    if layer > 4 and layer % 4:
                        switch = math.ceil(layer / 4)
                        layer = layer % 4
                    elif layer > 4:
                        switch = math.ceil(layer / 4)
                        layer = 4
                    else:
                        switch = 0
                    if layer % 2 == 0 and layer % 4 != 0:
                        for i in range(size):
                            if i % 2 or i + 2 > size:
                                continue
                            else:
                                for j in range(size):
                                    edge_list.append(((i, j), (i + 1, j)))
                        for l in range(size):
                            if switch % 2 and switch != 0 and l + 1 < size:
                                if l % 2:
                                    edge_list.append(((size - 1, l), (size - 1, l + 1)))
                            else:
                                if l % 2 == 0 and l + 1 < size:
                                    edge_list.append(((size - 1, l), (size - 1, l + 1)))
                    elif layer % 3 == 0:
                        for i in range(size):
                            for j in range(size):
                                if j % 2 and j + 1 < size:
                                    edge_list.append(((i, j), (i, j + 1)))
                                else:
                                    continue
                        for l in range(size):
                            if switch % 2 and switch != 0 and l + 1 < size:
                                if l % 2:
                                    edge_list.append(((l, 0), (l + 1, 0)))
                            else:
                                if l % 2 == 0 and l + 1 < size:
                                    edge_list.append(((l, 0), (l + 1, 0)))
                    elif layer % 4 == 0:
                        for i in range(size):
                            if i % 2 and i + 1 < size:
                                for j in range(size):
                                    edge_list.append(((i, j), (i + 1, j)))
                            else:
                                continue
                        for l in range(size):
                            if switch % 2 and switch != 0 and l + 1 < size:
                                if l % 2:
                                    edge_list.append(((0, l), (0, l + 1)))
                            else:
                                if l % 2 == 0 and l + 1 < size:
                                    edge_list.append(((0, l), (0, l + 1)))
                    else:
                        for i in range(size):
                            for j in range(size):
                                if j % 2 or j + 2 > size:
                                    continue
                                else:
                                    edge_list.append(((i, j), (i, j + 1)))
                        for l in range(size):
                            if switch % 2 and switch != 0 and l + 1 < size:
                                if l % 2:
                                    edge_list.append(((l, size - 1), (l + 1, size - 1)))
                            else:
                                if l % 2 == 0 and l + 1 < size:
                                    edge_list.append(((l, size - 1), (l + 1, size - 1)))
                else:
                    if layer > 4 and layer % 4:
                        layer = layer % 4
                    elif layer > 4:
                        layer = 4
                    if layer % 2 == 0 and layer % 4 != 0:
                        for i in range(size):
                            if i % 2 or i + 2 > size:
                                continue
                            else:
                                for j in range(size):
                                    edge_list.append(((i, j), (i + 1, j)))
                    elif layer % 3 == 0:
                        for i in range(size):
                            for j in range(size):
                                if j % 2 and j + 1 < size:
                                    edge_list.append(((i, j), (i, j + 1)))
                                else:
                                    continue
                    elif layer % 4 == 0:
                        for i in range(size):
                            if i % 2 and i + 1 < size:
                                for j in range(size):
                                    edge_list.append(((i, j), (i + 1, j)))
                            else:
                                continue
                    else:
                        for i in range(size):
                            for j in range(size):
                                if j % 2 or j + 2 > size:
                                    continue
                                else:
                                    edge_list.append(((i, j), (i, j + 1)))
                n = 0
                matrix_number = {}
                for i in range(size):
                    for j in range(size):
                        matrix_number[(i, j)] = n
                        n += 1
                entang_list = [(matrix_number[i[0]], matrix_number[i[1]]) for i in edge_list]
                c.add(gates.CZ(q[0], q[1]) for q in entang_list)
            size = qubits
            c.add(gates.RY(q, theta=0, trainable=True) for q in range(size))

    if entanglement == 'original_RX':
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
            c.add(gates.RX(q, theta=0, trainable=True) for q in range(size))

    if entanglement == 'original_RX_alternating':
        for l in range(1, p + 1):
            if size == 2:
                c.add(gates.CZ(0, 1))
            elif l%2:
                if size%2:
                    c.add(gates.CZ(q, q + 1) for q in range(0, size - 2, 2))
                else:
                    c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))
                c.add(gates.RY(q, theta=0, trainable=True) for q in range(size))
            else:
                if size % 2:
                    c.add(gates.CZ(q, q + 1) for q in range(1, size-1, 2))
                else:
                    c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))
                c.add(gates.RX(q, theta=0, trainable=True) for q in range(size))


    if entanglement == 'easy_Lie':
        for l in range(1, p + 1):
            if size%2:
                c.add(gates.CZ(q, q + 1) for q in range(0, size - 2, 2))
            else:
                c.add(gates.CZ(q, q + 1) for q in range(0, size - 1, 2))
            c.add(gates.RY(q, theta=0, trainable=True) for q in range(1, size - 1))
            c.add(gates.RX(q, theta=0, trainable=True) for q in range(1, size - 1))
            if size % 2:
                c.add(gates.CZ(q, q + 1) for q in range(1, size - 1, 2))
            else:
                c.add(gates.CZ(q, q + 1) for q in range(1, size - 2, 2))
            c.add(gates.RY(q, theta=0, trainable=True) for q in range(0, size))
            c.add(gates.RX(q, theta=0, trainable=True) for q in range(0, size))


    return c


def overlap_retriver(ansaz, true_vector):
    overlap = callbacks.Overlap(true_vector)
    ansaz.add(gates.CallbackGate(overlap))
    return ansaz
