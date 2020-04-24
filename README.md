# ML Playground
Random ML experiments that illustrate some common concepts.


### Algorithms

* __Graham Scan__: An stack-based algorithm that generates a convex hull around a set of points. An SVM is essentially a midplane dividing the line joining the convex hulls belonging to the two different classes. _Graham Scan_ can thus be used instead of a gradient-based optimization algorithm. 

* __Isomap__: An implementation of the _Isomap_ visualization algorithm that operates on geodesic distances. Uses an _sklearn_ utility function (based on Djikstra's + Priority Queue)to compute pairwise shortest distances from a given graph's adjacency-cum-distance matrix.


### Data Structures

* __Count-Min Sketch__: A probablistic data structure used in _Big Data_ domain applications where a lot of unique data is hashed to a lower dimensional space by multiple hash functions. Used in ML systems in applications like checking set membership etc.

* __Sparsity__: A test script that shows the memory effeciency of sparse matrices stored in compressed formats.


### Metrics

* __Wasserstein Distance__: A symmetric (as opposed to KL divergence) metric to compute distance between two distributions where the distributions have different probability space dimensionalities. Uses an iterative optimiation step described [by Michiel Stock](https://michielstock.github.io/OptimalTransport/). Used a lot in Computer Vision and Deep Learning (Wasserstein GAN).

* __L2 Regularization__: A fast and memory-efficient vectorized implementation of _L2 regularization_ that run on Numpy. Numpy has a fast C++ backend and for the same reason also bypasses the _Python GIL_. Thus, it is much faster than a looped version.


### ML Models 

* __Multi-Armed Bandits__: A test script that illustrates multi-armed bandits with updates based on the _Thomson Sampling_ procedure.

* __GMM__: Vectorized implementation of the Expectation-Maximization algorithm tested on simple Gaussian Mixture Models.
