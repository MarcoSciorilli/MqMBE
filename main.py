if __name__ == '__main__':

    import multiVQA as vq
    #vq.dataretriver.initialize_database('MaxCutData')
    kind = 'Multibase'
    data_to_read = ['instance', 'energy_ratio']
    for i in range(100):
        parameters_to_fix = {'kind': kind, 'instance': f'{i}'}
        print(vq.datamanager.read_data('MaxCutData', 'MaxCutData', data_to_read, parameters_to_fix))
    for i in range(6, 19):
        print(f"number of nodes:{i}")
        for j in range(6):
            print(f"number of layers:{j}")
            vq.dataretriver.VQE_evaluater(starting=0, ending=100, layer_number=j, nodes_number=i, optimization='COBYLA',
                          graph_list=None,
                          pick_init_parameter=None, random_graphs=False, entanglement='basic', multibase=True)


