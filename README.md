# Declarative Management of Machine Learning Experimentation Data

Currently, most ML workloads are executed in an “one-shot” and “ad-hoc” style via a variety of scripts, lacking declarative specifications and lacking storage of a declarative description of the executed workload.
Therefore, the execution of such tasks typically does not make use of historical data about related experiments and thereby lacks optimization potential and fails to achieve comparability and reproducibility.
To tackle this challenge, we propose the creation of a repository of declarative descriptions of machine learning experiments and their corresponding evaluation data. 
This will enable meta learning on this repository, e.g., to use the historical data to recommend features for new datasets, to find starting configurations for hyperparameter search, or to automatically compute baseline predictions for new tasks on existing datasets. 
Furthermore, we can also leverage the historical data to optimize the execution of new ML tasks, e.g. by re-using previously materialized intermediate results.

The envisioned system consists of three major components:
- a “repository for experimentation data”, which mostly comprises of model metadata and evaluation data from experiments
- a “model parallel training and evaluation component”, which can compute evaluations for declaratively specified model configurations and store the resulting evaluations in the repository
- a “configuration selection” component that proposes promising model configurations, using established ML techniques, database-style optimizations and meta learning approaches based on the historical data in the repository

The benefit from such system is two-fold; incremental pipelines and warm starting pipeline training on new datasets.

## Incremental Pipelines

## Warmstarting

## Main Component

### Feature Transformations

### Hyperparameter Search

## Research Direction

### Improvement in prediction quality

### Improvement in Runtime

## Conclusion

