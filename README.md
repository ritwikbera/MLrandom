# ML Playground
Random ML experiments that illustrate some common concepts.

* __Graham Scan__: An stack-based algorithm that generates a convex hull around a set of points. An SVM is essentially a midplane dividing the line joining the convex hulls belonging to the two different classes. _Graham Scan_ can thus be used instead of a gradient-based optimization algorithm. 

* __GMM__: Vectorized implementation of the Expectation-Maximization algorithm tested on simple Gaussian Mixture Models.

* __L2 Regularization__: A fast and memory-efficient vectorized implementation of _L2 regularization_ that run on Numpy. Numpy has a fast C++ backend and for the same reason also bypasses the _Python GIL_. Thus, it is much faster than a looped version.

* __Multi-Armed Bandits__: A test script that illustrates multi-armed bandits with updates based on the _Thomson Sampling_ procedure.

* __Sparsity__: A test script that shows the memory effeciency of sparse matrices stored in compressed formats.
