if __name__ == '__main__':

    import mqMBE as vq
    import numpy as np
    vq.dataretriver.Benchmarker.initialize_database('Test')
    vq.dataretriver.Benchmarker(database_name='Test', optimization='SLSQP', entanglement='rotating',layer_number=[6], compression=2, multiprocessing=False, activation_function=np.tanh, hyperparameters=[1,0.4])
