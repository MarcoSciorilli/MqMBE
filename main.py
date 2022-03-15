if __name__ == '__main__':

    import exploVQE as vq

    print("ARRIVO QUI")
    for i in range(4, 25):
        print(f"number of nodes:{i}")
        vq.dataretriver.VQE_evaluater(starting=0, ending=100, layer_number=1, nodes_number=6, optimization='COBYLA',
                      graph_list=None,
                      pick_init_parameter=True, random_graphs=True, entanglement='basic', multibase=True)


