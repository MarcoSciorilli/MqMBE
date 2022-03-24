if __name__ == '__main__':

    import multiVQA as vq
    def nodes_compressed(quibits):
        return(3*(quibits**2+quibits)/2)
    def max_compression(quibits):
        return 4**quibits-1
    #vq.dataretriver.initialize_database('MaxCutDatabase')
    # for i in range(14, 19):
    #     vq.dataretriver.benchmarker(starting=0, ending=100, nodes_number=i, kind='bruteforce')

    # for i in range(4, 20):
    #     print(f'Nodes number:{i}')
    #     vq.dataretriver.benchmarker(starting=0, ending=100, trials= 5, nodes_number=i, kind='goemans_williamson')
    #
    m = 2
    for i in range(8, 103):
        print(f'Nodes number:{i}', m)
        if i > nodes_compressed(m):
            m += 1
        if i != nodes_compressed(m):
            continue
        for j in range(7):
            print(f'Layer number:{j}')
            vq.dataretriver.benchmarker(starting=0, ending=100, trials=5, nodes_number=i, kind='multibaseVQA',
                                        layer_number=j, optimization='SLSQP',
                                        compression=nodes_compressed(m) / max_compression(m), pauli_string_length=m,
                                        entanglement='basic')


    m = 2
    for i in range(8, 103):
        print(f'Nodes number:{i}', m)
        if i > nodes_compressed(m):
            m += 1
        if i != nodes_compressed(m):
            continue
        for j in range(7):
            print(f'Layer number:{j}')
            vq.dataretriver.benchmarker(starting=0, ending=100, trials=5, nodes_number=i, kind='multibaseVQA',
                                        layer_number=j, optimization='Powell',
                                        compression=nodes_compressed(m) / max_compression(m), pauli_string_length=m,
                                        entanglement='basic')

    m = 2
    for i in range(8, 103):
        print(f'Nodes number:{i}', m)
        if i > nodes_compressed(m):
            m += 1
        if i != nodes_compressed(m):
            continue
        for j in range(7):
            print(f'Layer number:{j}')
            vq.dataretriver.benchmarker(starting=0, ending=100, trials=5, nodes_number=i, kind='multibaseVQA',
                                        layer_number=j, optimization='COBYLA',
                                        compression=nodes_compressed(m) / max_compression(m), pauli_string_length=m,
                                        entanglement='basic')