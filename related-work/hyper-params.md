## Efficient and Robust Automated Machine Learning
In this work, the authors offer a method of automatically solving machine learning problems on new datasets by using information on existing datasets.
The workflow is as follows:
- For all the existing datasets, run bayesian hyperparameter optimization in offline mode. Store the most promising setting for each dataset
- Given a new dataset, find the closest 25 datasets (L1 distance) and use those hyperparameters as starting point for another round of hyperparameter optimization
- Complement with an ensemble building step. Reuse the computation in hyperparameter optimization step to speed up the ensemble creation
