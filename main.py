if __name__ == '__main__':

    import exploVQE as vq


    for j in range(2, 7):
        print(f"Layers: {j}")
        for i in range(4, 14):
            print(f"Quibits: {i}")
            vq.dataretriver.overlap_evaluater(starting=0, ending=100, layer_number=j, nodes_number=i,
                                            initial_point=True, random=True, entanglement='basic', optimization='cma')


    for j in range(1, 7):
        print(f"Layers: {j}")
        for i in range(4, 14):
            print(f"Quibits: {i}")
            vq.dataretriver.overlap_evaluater(starting=0, ending=100, layer_number=j, nodes_number=i,
                                            initial_point=True, random=True, entanglement='circular', optimization='COBYLA')

    for j in range(1, 7):
        print(f"Layers: {j}")
        for i in range(4, 14):
            print(f"Quibits: {i}")
            vq.dataretriver.overlap_evaluater(starting=0, ending=100, layer_number=j, nodes_number=i,
                                            initial_point=True, random=True, entanglement='circular-interleaved', optimization='COBYLA')