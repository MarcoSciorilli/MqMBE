import math

if __name__ == '__main__':

    import multiVQA as vq
    import numpy as np
    import os
    import scipy.optimize

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from multiVQA.datamanager import read_data
    def nodes_compressed(quibits):
        return int((3 * (quibits ** 2 + quibits) / 2))


    def max_compression(quibits):
        return 4 ** quibits - 1


    import networkx as nx
    # vq.dataretriver.Benchmarker.initialize_database('MaxCutDatabase_100_SLSQP_rotating_full')



    graph_dict = {}
    # graph_dict["toruspm3-8-50.dat"] = (nx.read_weighted_edgelist("toruspm3-8-50.dat"))
    graph_dict["w09_100.0"] = (nx.read_weighted_edgelist("w09_100.0"))
    graph_dict["w09_100.1"] = (nx.read_weighted_edgelist("w09_100.1"))
    graph_dict["w09_100.2"] = (nx.read_weighted_edgelist("w09_100.2"))
    #
    vq.dataretriver.Benchmarker(starting=10, ending=11, trials=20, graph_dict=graph_dict, nodes_number=100,
                                kind='multibaseVQA',
                                layer_number=list(range(2,30,2)), optimization='SLSQP', compression=2,
                                entanglement='rotating_full', activation_function=np.tanh, hyperparameters=[1.5, 2],
                                database_name='MaxCutDatabase_100_SLSQP_rotating_full', multiprocessing=True)

    # vq.dataretriver.Benchmarker(starting=0, ending=100, trials=20, nodes_number=512,
    #                                 kind='BURER2002',  graph_dict=graph_dict, database_name='MaxCutDatabase_512_bur')
    # for i in range(14, 19):
    #     vq.dataretriver.Benchmarker(starting=0, ending=100, nodes_number=i, kind='bruteforce')

    # for i in range(4, 85):
    #     print(f'Nodes number:{i}')
    # vq.dataretriver.Benchmarker(starting=0, ending=100, trials=20, nodes_number=100,
    #                                 kind='BURER2002',  graph_dict=graph_dict, database_name='MaxCutDatabase_bur')
    # for k in [4,5]:
    #     for i in range(10, 19):
    #         print(f'Nodes number:{i}')
    #         for j in range(0, 5):
    #             print(f'Layer number:{j}')
    #             vq.dataretriver.Benchmarker(starting=0, ending=100, trials=5, nodes_number=i, kind='multibaseVQA',
    #                                         layer_number=j, optimization='SLSQP', compression=2,
    #                                         entanglement='article',
    #                                         activation_function=np.tanh, hyperparameters=[1.5, 1], shuffle=True, qubits=k)
    # vq.dataretriver.Benchmarker(starting=0, ending=100, trials=5, nodes_number=63, kind='multibaseVQA',
    #                             layer_number=list(range(0, 11)), optimization='SLSQP', compression=2,
    #                             entanglement='article', activation_function=np.tanh, hyperparameters=[1.5, 2],
    #                             graph_kind='fully')

    # for i in [ 550]:
    #     print(f'Nodes number:{i}')
    #     vq.dataretriver.Benchmarker(starting=0, ending=100, trials=5, nodes_number=i, kind='multibaseVQA',
    #                                 layer_number=list(range(5)), optimization='COBYLA', compression=2,
    #                                 entanglement='simple', activation_function=np.tanh, hyperparameters=[1.5, 2], precision='single', database_name='Test', multiprocessing=False)
    # for i in [9 ]:
    #     vq.dataretriver.Benchmarker(starting=0, ending=10, trials=5, nodes_number=i, kind='multibaseVQA',
    #                                 layer_number=list(range(1, 6)), optimization='COBYLA', compression=2,qubits=4,
    #                                 entanglement='rotating', activation_function=np.tanh, hyperparameters=[1.5,2], database_name='MaxCutDatabase_COBYLA_rotating', precision='single')

    # def fine_tuner(hyperparameters, layers, nodes):
    #     def func(hyperparameters, layers, nodes):
    #         vq.dataretriver.Benchmarker(starting=0, ending=100, trials=5, nodes_number=nodes, kind='multibaseVQA',
    #                                             layer_number=layers, optimization='SLSQP', compression=2,
    #                                             entanglement='article',
    #                                             activation_function=np.tanh, hyperparameters=hyperparameters, shuffle=True,  same_letter=False)
    #         cuts = read_data('MaxCutDatabase', 'MaxCutDatabase', ['max_energy'], {'layer_number':layers, 'hyperparameter':str(hyperparameters)})
    #         loss = -np.sum(cuts)
    #         print(loss, hyperparameters)
    #         return loss
    #     scipy.optimize.minimize(func, hyperparameters, args=(layers, nodes), method='COBYLA')
    #
    # fine_tuner([1.5,2], 3, 9)

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
