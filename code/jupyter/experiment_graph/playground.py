# import openml stuff

from experiment_graph.execution_environment import ExecutionEnvironment
import pandas as pd
from openml import runs, flows, setups, evaluations, tasks, datasets

execution_environment = ExecutionEnvironment()

# load openml dataset by id (unique identifier) into experiment experiment_graphs

# for a given run id
#   load the data
#   get the flow
#   get individual components of the flow as scikit learn operations
#   add the to the experiment_graphs
#   execute and update the experiment_graphs
from openml_connectors import *

download_dataset(task_id=31, root_path='../data/openml', overwrite=True)


from sklearn_helper.sklearn_wrappers import LabelEncoder