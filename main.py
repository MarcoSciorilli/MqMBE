if __name__ == '__main__':

    import multiVQA as vq
    # for i in range(4,19):
    #     for j in range(8):
    #         vq.dataretriver.benchmarker(starting=0, ending=100, nodes_number=i, kind='multibaseVQA', layer_number=j, optimization= 'COBYLA',  compression= 1, pauli_string_length = 1, entanglement = 'basic')
    #
    # for i in range(4,19):
    #     for j in range(8):
    #         vq.dataretriver.benchmarker(starting=0, ending=100, nodes_number=i, kind='multibaseVQA', layer_number=j, optimization= 'COBYLA',  compression= 2/3, pauli_string_length = 1, entanglement = 'basic')
    #
    # for i in range(4,19):
    #     for j in range(8):
    #         vq.dataretriver.benchmarker(starting=0, ending=100, nodes_number=i, kind='multibaseVQA', layer_number=j, optimization= 'COBYLA',  compression=  1/3, pauli_string_length = 1, entanglement = 'basic')

    for i in range(4,19):
        for j in range(8):
            vq.dataretriver.benchmarker(starting=0, ending=100, nodes_number=i, kind='multibaseVQA', layer_number=j, optimization= 'COBYLA',  compression= 1, pauli_string_length = 2, entanglement = 'basic')