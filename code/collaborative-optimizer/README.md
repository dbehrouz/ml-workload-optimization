# Collaborative Workload Optimizer

This readme file contains instructions for setting up and running the example scripts.

## Setup 
There's no pip installation yet. 
To use the collaborative optimizer, clone the repository and update the PYTHONPATH to include the path to the current folder. For example:
```
export PYTHONPATH="${PYTHONPATH}:/path-to-where-the-repository-is-cloned/ml-workload-optimization/code/collaborative-optimizer
```

Required packages:
```
pip install -U pandas 
pip install -U scikit-learn 
pip install -U networkx 
pip install -U kaggle 
pip install -U matplotlib 
pip install -U seaborn 
pip install -U pympler 
pip install -U lightgbm 
pip install -U 'openml==0.8.0'
```

## Examples
To use the optimized execution environment instead of pandas use the following import and load csv files:
```python
from experiment_graph.execution_environment import ExecutionEnvironment
execution_environment = ExecutionEnvironment()
data = execution_environment.load('path-to-data/data.csv')
```

There is support for several sklearn operators and models which can be found [here](experiment_graph/sklearn_helper)

You can find a simple end-to-end example notebook [here](examples/).

There are several other scripts [here](paper/experiment_workloads/kaggle_home_credit/optimized) as well.

## Advanced Usage
To pass custom scikit learn models (or any model that follows the fit/transform API of scikit learn), you can do the following:
```python
from experiment_graph.execution_environment import ExecutionEnvironment
execution_environment = ExecutionEnvironment()
data = execution_environment.load('path-to-data/data.csv')


from sklearn.impute import SimpleImputer # Actual scikit learn model
sk_imputer = SimpleImputer()
trained_imputer = data.fit_sk_model(sk_imputer)
# call trained_imputer.data() to get the actual trained scikit learn model
imputed_data = trained_imputer.transform(data)
print(imputed_data.data())
labels = execution_environment.load('path-to-labels/labels.csv')

from sklearn.ensemble import RandomForestClassifier
sk_random_forest_model = RandomForestClassifier()
trained_random_forest = imputed_data.fit_sk_model_with_labels(sk_random_forest_model, labels)
# call trained_random_forest.data() to get the actual trained scikit learn model
test_data = execution_environment.load('path-to-data/test_data.csv')
test_labels = execution_environment.load('path-to-data/test_labels.csv')
score = trained_random_forest.score(test_data, test_labels)
print(score.data())
``` 

Check [node.py](experiment_graph/graph/node.py) for a full list of available methods for the following entities:
```
Dataset: equivalent to the pandas DataFrame, execution_environment.load returns a Dataset
Feature: equivalent to the pandas DataSeries
SK_Model: wrapper for models. the resut of Dataset.fit_sk_model_with_labels or Dataset.fit_sk_model are SK_model. 
Agg: result of aggregate operations such as sum, nunique, min, and max on Dataset and Feature
SuperNode: Special nodes for representing multi-input operators such as merge and concat    
```





