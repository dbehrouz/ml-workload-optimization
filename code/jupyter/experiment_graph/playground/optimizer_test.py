"""
Checking the Reuse and Warm-starting correctness
Use Cases:
UC 1:
    1. Load data
    2. chain 2 data transformation and a model training
    3. store the graph
    4. open a new workload
    5. load the history graph
    6. write the exact same transformations and model training
    7. the model should be fetched from the history graph instead being executed
UC 2:
    1. Load data
    2. chain 2 data transformation and a model training
    3. store the graph
    4. open a new workload
    5. load the history graph
    6. write the exact same transformations but a different model training procedure
    7. the data after the second transformation should be fetched from the history and only the model training
       should be executed

UC 3:
    workloads with forks and merges

UC 4:
    (After implementing the concept of materialization algorithms)
    history graph with partial materialized nodes
"""
from execution_environment import ExecutionEnvironment


def usecase_1():
    ee = ExecutionEnvironment('dedup')
    ROOT_PACKAGE_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter'
    root_data = ROOT_PACKAGE_DIRECTORY + '/data'
    DATABASE_PATH = root_data + '/optimizer_test/uc1'

    train = ee.load(root_data + '/openml/task_id=31/train.csv')


