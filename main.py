if __name__ == '__main__':

    import multiVQA as vq
    import numpy as np


    def nodes_compressed(quibits):
        return int((3 * (quibits ** 2 + quibits) / 2))


    def max_compression(quibits):
        return 4 ** quibits - 1


    vq.dataretriver.Benchmarker.initialize_database('MaxCutDatabase')
    # for i in range(14, 19):
    #     vq.dataretriver.Benchmarker(starting=0, ending=100, nodes_number=i, kind='bruteforce')

    # for i in range(4, 85):
    #     print(f'Nodes number:{i}')
    #     vq.dataretriver.Benchmarker(starting=0, ending=100, trials= 5, nodes_number=i, kind='goemans_williamson')

    m = 5
    for i in range(44, 46):
        print(f'Nodes number:{i}', m)
        print(i, nodes_compressed(m))
        if i > nodes_compressed(m):
            m += 1
        if i != nodes_compressed(m):
            continue
        for j in range(0, 21, 5):
            print(f'Layer number:{j}')
            vq.dataretriver.Benchmarker(starting=0, ending=100, trials=5, nodes_number=i, kind='multibaseVQA',
                                        layer_number=j, optimization='SLSQP',
                                        ratio_total_words=nodes_compressed(m) / max_compression(m),
                                        pauli_string_length=m,
                                        entanglement='article',
                                        activation_function=vq.multibase.MultibaseVQA.my_activation)

    # for i in [9, 18, 30, 45]:
    #     for j in range(7):
    #         print(f'Layer number:{j}')
    #         vq.dataretriver.Benchmarker(starting=0, ending=100, trials=5, nodes_number=i, kind='multibaseVQA',
    #                                     layer_number=j, optimization='SLSQP',
    #                                     ratio_total_words=1, pauli_string_length=1, activation_function=np.tanh,
    #                                     entanglement='article')
    #
    # m = 2
    # for i in range(8, 46):
    #     print(f'Nodes number:{i}', m)
    #     if i > nodes_compressed(m):
    #         m += 1
    #     if i != nodes_compressed(m):
    #         continue
    #     for j in range(7):
    #         print(f'Layer number:{j}')
    #         vq.dataretriver.benchmarker(starting=0, ending=100, trials=5, nodes_number=i, kind='multibaseVQA',
    #                                     layer_number=j, optimization='cma',
    #                                     ratio_total_words=nodes_compressed(m) / max_compression(m), pauli_string_length=m,
    #                                     entanglement='basic')
