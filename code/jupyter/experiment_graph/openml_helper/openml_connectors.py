import os

import pandas as pd
import numpy as np
from openml import tasks, datasets, runs, flows, setups

from execution_environment import ExecutionEnvironment

OPENML_ROOT_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter/data/openml'


def download_dataset(task_id, root_path, overwrite=False):
    """
    for ease of use, we always store the datasets locally
    we store the datasets with the same openml id so it is easier to distinguish
    :param overwrite: if the existing data should be rewritten
    :param root_path: root path to store the dataset as pandas csv frame
    :param task_id: open ml id of the dataset
    """
    task = tasks.get_task(task_id=task_id)
    dataset = datasets.get_dataset(dataset_id=task.dataset_id)
    data, columns = dataset.get_data(return_attribute_names=True)
    train_indices, test_indices = task.get_train_test_split_indices()
    train = pd.DataFrame(data=data[train_indices], columns=columns)
    test = pd.DataFrame(data=data[test_indices], columns=columns)
    result_path = root_path + '/task_id=' + str(task.dataset_id)
    if os.path.exists(result_path) and not overwrite:
        raise Exception('Directory already exists and overwrite is not allowed')

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    train.to_csv(result_path + '/train.csv', index=False)
    test.to_csv(result_path + '/test.csv', index=False)


def parseValue(value):
    import ast
    try:
        if (value == 'null'):
            return None
        if (value == 'true'):
            return True
        if (value == 'false'):
            return False
        actual = ast.literal_eval(value)
        if type(actual) is list:
            return sorted(actual)
        if type(actual) is str:
            return actual.replace('"', '')

        return actual
    except:
        return value


def flow_to_edge_list(flow, setup):
    edges = []
    for componentKey, componentValue in flow.components.items():
        prefix = componentKey
        fullName = componentValue.class_name
        componentParams = dict()
        for paramKey, paramValue in setup.parameters.items():
            if paramValue.full_name.startswith(fullName):
                # Openml saves the type informatino in a weird way so we have to write a special piece of code
                if paramValue.parameter_name == 'dtype':
                    componentParams[str(paramValue.parameter_name)] = np.float64
                    # typeValue = self.parseValue(paramValue.value)['value']
                    # if (typeValue == 'np.float64'):
                    #    componentParams[str(paramValue.parameter_name)] = np.float64
                    # else:
                    #    componentParams[str(paramValue.parameter_name)] = typeValue
                elif paramValue.parameter_name == 'random_state':
                    componentParams[str(paramValue.parameter_name)] = 14766
                else:
                    componentParams[str(paramValue.parameter_name)] = parseValue(paramValue.value)
        edges.append((prefix, fullName, componentParams))
        # comp = Component(prefix, fullName, componentParams)
        # experimentObject.components.append(comp)
    return edges


def skpipeline_to_edge_list(pipeline, setup):
    def getFullyQualifiedName(o):
        return o.__module__ + "." + o.__class__.__name__

    edges = []
    for componentKey, componentValue in pipeline.steps:
        prefix = componentKey
        fullName = getFullyQualifiedName(componentValue)
        componentParams = dict()
        for paramKey, paramValue in setup.parameters.items():
            if paramValue.full_name.startswith(fullName):
                # Openml saves the type informatino in a weird way so we have to write a special piece of code
                if paramValue.parameter_name == 'dtype':
                    componentParams[str(paramValue.parameter_name)] = np.float64
                    # typeValue = self.parseValue(paramValue.value)['value']
                    # if (typeValue == 'np.float64'):
                    #    componentParams[str(paramValue.parameter_name)] = np.float64
                    # else:
                    #    componentParams[str(paramValue.parameter_name)] = typeValue
                elif paramValue.parameter_name == 'random_state':
                    componentParams[str(paramValue.parameter_name)] = 14766
                else:
                    componentParams[str(paramValue.parameter_name)] = parseValue(paramValue.value)
        edges.append((prefix, fullName, componentParams))
    return edges


def run_to_workload(run_id):
    """
    processes the given run and returns the workload representation of ti
    :param run_id: run id
    :return: workload represetation in graph form
    """
    run = runs.get_run(run_id)
    execution_evironment = ExecutionEnvironment('dedup')
    # TODO check if the dataset exists or not
    train_data = execution_evironment.load(OPENML_ROOT_DIRECTORY + '/task_id=' + str(run.task_id) + '/train.csv')

    flow = flows.get_flow(run.flow_id)
    flow.dependencies = u'sklearn>=0.19.1\nnumpy>=1.6.1\nscipy>=0.9'
    for v in flow.components.itervalues():
        v.dependencies = u'sklearn>=0.19.1\nnumpy>=1.6.1\nscipy>=0.9'
    pipeline = flows.flow_to_sklearn(flow)
    setup = setups.get_setup(run.setup_id)
    edges = skpipeline_to_edge_list(pipeline, setup)
    return edges


print run_to_workload(5768599)
