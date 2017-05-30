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

## Experiment Database Schema
![Data Model](images/datamodel-crop.png)

## Research Direction
The proposed database improves machine learning pipeline creation in two different ways.
First, by gathering statistics from the history, we are able to recommend users what are the best set of transformations and models for specific datasets.
Second, by defining each datasets by a set of properties we are able to extend this recommendation to new datasets.
New datasets are examined against the database and the most similar or set of similar datasets are chosen. 
Based on the machine learning task at hand, different transformation and machine learning models are recommended to the user.

We expect an improvement both in the quality and time of the created pipeline.

### Improvement in prediction quality
When encountered with new datasets and new machine learning tasks, users have limited budget to explore all the available options to create a pipeline.
As a result, the search for the pipeline that provides the highest prediction quality ends before the desired pipeline is reached.
By exploiting the information in the experiment database, the search space of the available options is drastically removed allowing the users to focus on more promising configurations for their pipeline, hence increasing the chance of attaining a pipeline that provides a higher quality.
This has been the topic of many of the existing work in the machine learning community [1,2].

### Improvement in time
Our main focus is open improvements in time.
The obvious benefit of such database comes from the recommendations that it provides.
Users will spend less time examining all the available configurations for the task at hand and will use the recommended configuration to create a pipeline.
We propose two other improvements that will reduce both the human latency and execution time.
**Incremental processing of pipelines:** 
Real world machine learning applications typically include a pipeline that is created over an initial dataset. 
However, as the application is running, new data becomes available. 
The new data is processed periodically and the pipeline is improved based on the new data.
Since the format of the newly available data is the same as the existing dataset, the same set of transformations can be applied without much manual intervention.
However, many of these transformations have internal parameters that are dependent on the historical data.
By using the information from the experiment database, we avoid the reprocessing of the entire historical data and can directly apply the latest versions of the transformations to the new batch of data.

**Materialization of common transformations: **
By tracking the usage of the users of different datasets and transformations, we can materialized the more common transformations and as a result speed up the execution time of the pipeline.

## Incremental Pipelines

## Main Component

### Feature Transformations

### Hyperparameter Search

## Designing the Experiments

## Conclusion

## References
[1] Feurer, Matthias, et al. "Efficient and robust automated machine learning." Advances in Neural Information Processing Systems. 2015.
[2] Vanschoren, Joaquin, et al. "OpenML: networked science in machine learning." ACM SIGKDD Explorations Newsletter 15.2 (2014): 49-60.