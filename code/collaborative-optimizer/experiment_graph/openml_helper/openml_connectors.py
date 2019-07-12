import numpy as np
import pandas as pd
from openml import tasks, datasets, runs, flows, setups, config

from execution_environment import ExecutionEnvironment

OPENML_ROOT_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/data/openml'


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


def parse_value(value):
    import ast
    try:
        if value == 'null':
            return None
        if value == 'true':
            return True
        if value == 'false':
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
                # Openml saves the type information in a weird way so we have to write a special piece of code
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
                    componentParams[str(paramValue.parameter_name)] = parse_value(paramValue.value)
        edges.append((prefix, fullName, componentParams))
        # comp = Component(prefix, fullName, componentParams)
        # experimentObject.components.append(comp)
    return edges


def skpipeline_to_edge_list(pipeline, setup):
    def get_fully_qualified_name(o):
        return o.__module__ + "." + o.__class__.__name__

    def relaxed_match(sklearn_class, openml_side):
        if sklearn_class.__class__.__name__ in openml_side:
            return True
        else:
            return False

    edges = []
    for componentKey, componentValue in pipeline.steps:
        # prefix = componentKey
        fullName = get_fully_qualified_name(componentValue)
        componentParams = dict()
        for paramKey, paramValue in setup.parameters.items():
            if paramValue.parameter_name in componentValue.get_params().keys():
                # Openml saves the type information in a weird way so we have to write a special piece of code
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
                    componentParams[str(paramValue.parameter_name)] = parse_value(paramValue.value)
        componentValue.set_params(**componentParams)
        # edges.append((prefix, componentValue, componentParams))
        edges.append(componentValue)
    return edges


def hack_dependency(dependency_string, component):
    component.dependencies = dependency_string
    return component


def repair(flow):
    """
    Flows 5981, 5987, 6223 use custom class: helper.dual_imputer.DualImputer which we mark is unrepairable
    repair a flow and return it back
    :param flow:
    :return:
    """
    flow.dependencies = u'sklearn>=0.19.1\nnumpy>=1.6.1\nscipy>=0.9'
    for v in flow.components.itervalues():
        if v.class_name.startswith('helper.dual_imputer') or v.class_name.startswith('extra.dual_imputer'):
            print 'undefined class: {}'.format(v.class_name)
            return 'ERROR'
        # if 'OneHotEncoder' in v.class_name:
        #     v.class_name = 'sklearn.preprocessing.OneHotEncoder'
        v.dependencies = u'sklearn>=0.19.1\nnumpy>=1.6.1\nscipy>=0.9'
        # for meta components (e.g, boosting algorithms)
        if hasattr(v, 'components'):
            for vv in v.components.itervalues():
                vv.dependencies = u'sklearn>=0.19.1\nnumpy>=1.6.1\nscipy>=0.9'

    return flow


def run_to_workload(run_id, execution_environment):
    """
    main function running an openml run with the graph.
    TODO
        currently it runs the graph at the end of function. We should make the optimization model
        which gets the graph of this run and the existing experiment graph
        and run the optimized version on return

    :param execution_environment:
    :param run_id: run id
    :return: workload represetation in graph form
    """
    try:
        run = runs.get_run(run_id)
        # execution_environment = ExecutionEnvironment('dedup')
        # TODO check if the dataset exists or not
        train_data = execution_environment.load(OPENML_ROOT_DIRECTORY + '/task_id=' + str(run.task_id) + '/train.csv')

        y = train_data['class']
        x = train_data.drop('class')
        flow = repair(flows.get_flow(run.flow_id))
        if flow == 'ERROR':
            print 'skipping run {} because flow {} is not repairable'.format(run_id, run.flow_id)
            return
        pipeline = flows.flow_to_sklearn(flow)
        setup = setups.get_setup(run.setup_id)
        edges = skpipeline_to_edge_list(pipeline, setup)
        # print edges
        # print 'run : {} flow : {}'.format(run_id, run.flow_id)
        for i in range(len(edges) - 1):
            model = x.fit_sk_model(edges[i])
            x = model.transform(x)

        # TODO is it OK to assume all the openml pipelines end with a scikit learn model?
        model = x.fit_sk_model_with_labels(edges[-1], y)

        model.data()
    except Exception:
        print ('ERROR AT FLOW:{} and RUN: {}'.format(flow.flow_id, run_id))


import os

config.cache_directory = os.path.expanduser(OPENML_ROOT_DIRECTORY + '/cache')
flow_list = pd.DataFrame.from_dict(flows.list_flows(), orient='index')
allowed = flow_list[flow_list.full_name.str.startswith('sklearn.pipeline')].id
n = len(allowed) / 100
run_list = []
for i in range(n):
    run_list.extend(runs.list_runs(size=10000, task=[31], flow=allowed[(i * 100):(i + 1) * 100]).keys())
#
execution_environment = ExecutionEnvironment('dedup')
for r in run_list[:100]:
    run_to_workload(r, execution_environment)
# run_to_workload(2081539, execution_environment)

import matplotlib.pyplot as plt

execution_environment.workload_graph.plot_graph(plt, vertex_freq=True, edge_oper=True, edge_time=True)
plt.show()
