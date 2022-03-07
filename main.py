if __name__ == '__main__':

    import exploVQE as vq
    for j in range(1, 7):
        print(f"Layers: {j}")
        for i in range(4, 16):
            print(f"Quibits: {i}")
            vq.dataretriver.overlap_evaluater(starting=0, ending=100, layer_number=j, nodes_number=i,
                                            initial_point=True, random=True)

    for j in range(1, 7):
        print(f"Layers: {j}")
        for i in range(4, 15):
            print(f"Quibits: {i}")
            vq.dataretriver.overlap_evaluater_parallel(starting=0, ending=100, layer_number=j, nodes_number=i,
                                            initial_point=True, random=True)
