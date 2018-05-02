## Efficient and Robust Automated Machine Learning [1]
In this work, the authors offer a method of automatically solving machine learning problems on new datasets by using information on existing datasets.
The workflow is as follows:
- For all the existing datasets, run bayesian hyperparameter optimization in offline mode. Store the most promising setting for each dataset
- Given a new dataset, find the closest 25 datasets (L1 distance) and use those hyperparameters as starting point for another round of hyperparameter optimization
- Complement with an ensemble building step. Reuse the computation in hyperparameter optimization step to speed up the ensemble creation

Datasets are described using various features. 
The full list can be found here: 
http://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-supplementary.pdf
In this work, they mostly used landmark classifiers as features of datasets and computed the distance using those.

## Collaborative hyperparameter tuning [2]
The authors propose a method for hyperparameter tuning across different datasets.
They redefine the problem of SMBO and introduced an extra input, the dataset, to the problem.
Datasets are defined in terms of the number of features, classes, instances (in some experiments also skewness and kurtosis). 
The SMBO process now searches in a space of D (datasets) and H (hyperparameters) simultaneously. 
The idea is that, if for dataset "A" the hyperparameter setting "X" performs better than "Y", it is likely that "X" will perform than "Y" for dataset "B" as well. 
Based on this idea, the new SMBO algorithm computes this ranking for every dataset and use the information for other datasets.


## Efficient Transfer Learning Method for Automatic Hyperparameter Tuning [3]
In this work, the authors offer an efficient method for using hyperparameters for one problem to guide the search for a new problem.
Unlike [2], the focus of this work is on a more realistic workflow, where problems and datasets arrive in a stream and have to solve one at a time.
Therefore, previous datasets cannot be stored and revisited.
Their method is more efficient than[2].
They propose a new kernel that involves two datasets (based on the distance in the feature space) (called multiple kernel framework MLK).
They only used three features to describe datasets, number of classes, features, and instances. 
Their method (especially the MLK) performs better than Random search, separate GPs and [2] (in most cases).




# References
[1] Feurer, Matthias, et al. "Efficient and robust automated machine learning." Advances in Neural Information Processing Systems. 2015.
[2] Bardenet, Rémi, et al. "Collaborative hyperparameter tuning." International Conference on Machine Learning. 2013.
[3] Yogatama, Dani, and Gideon Mann. "Efficient transfer learning method for automatic hyperparameter tuning." Artificial Intelligence and Statistics. 2014.
