# MqMBE

MqMBE is a package built to benchmark an extension to the Multi-basis encoding proposed to solve the weighted MaxCut problem using VQA (**[ arXiv:2106.13304](https://arxiv.org/abs/2106.13304)**).

This extension allows the user to use quadratically less qubits in the circuit architecture, encoding the graph nodes in the two-body correlator observables of the same pauli letter. 

This library is built on the QIBO framework for the simulation of quantum circuit.

## Dependencies 

The package rely on the following not-standard python packages: qibo, numpy, tensorflow, MQLib, networkx, cvxpy, sqlite3.

## Modules

* ansatz : module implementing the circuits architectures.
* datamanader : naive implementation of the utility functions necessary for a SQLite table.
* dataretriver: module implementing the benchmark itself, allowing the user gather data with a single method call.
* multibase: module implementing the encoding itself.
* newgraph: module which generate instances of weighted MaxCut problems.
* plots: module containing function useful to plot out the results (still in the making)
* resultevaluater:  module containing few methods to postprocess the results of the algorithm.



## Demo

Before starting the benchmark, it is necessary to initialize a database in which the benchmark results is going to be stored. This can be done with the method:

```
vq.dataretriver.Benchmarker.initialize_database('Name_of_the_database')
```

After that,  the user can run the benchmark with the command Benchmark. As this method is built to test both classic and quantistic algorithms (which needs very different setup) most of the parameters are None by default. For a quick test of the MqMBE, we suggest the following initialization of the parameters:

```
 vq.dataretriver.Benchmarker(database_name='Name_of_the_database', optimization='SLSQP', entanglement='rotating',layer_number=[6], compression=2, multiprocessing=False, activation_function=np.tanh, hyperparameters=[1,0.4])

```

Details about all the possible parameters of Benchmarker() can be found in the dataretriver module.
