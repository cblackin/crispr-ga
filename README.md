# CRISPR Enhanced Genetic Algorithm

The CRISPR Enhanced Genetic Algorithm is a proof of concept for an evolutionary algorithm based on the CRISPR Cas9 gene editing system. 

CRISPR GA implements two operators to (1) track beneficial sequences among the fittest individuals (analagous to building blocks) and (2) perform guided insertion of the sequences on offspring. Guided insertion will select a sequence from among the most recent generations based off its observed frequency. 

An implementation of CRISPR GA to the TSP is ongoing, and will involve comparisons with other advanced GAs such as a hybrid/memetic GA, and GOMEA.

A [notebook](src/knapsack-experiments.ipynb) has been provided with example configurations for a GA, an Estimation of Distribution Algorithm (EDA), a Gene-pool Optimal Mixing Algorithm (GOMEA) and CRISPR GA on the knapsack problem. Note that the GOMEA algorithm implementation is not finalized, and it was not included in the conference paper.