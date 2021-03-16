import pickle
import errno
import os

import numpy as np
import pandas as pd
from openml import tasks, datasets, runs, flows, setups, config

OPENML_ROOT_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/data/openml'
FLOW_DICTIONARY = 'flows.pickle'
SETUP_DICTIONARY = 'setups.pickle'
SETUP_FLOW_LIST = 'setup_flow.pickle'
FILTERED_SETUP = 'filtered_setups.csv'
EXCLUDE_FLOWS = [5981, 5987, 5983, 5981, 6223]


def download_dataset(openml_dir, task_id):
    """
    for ease of use, we always store the datasets locally
    we store the datasets with the same openml id so it is easier to distinguish

    :param openml_dir: root openml directory
    :param task_id: open ml id of the dataset
    """
    task = tasks.get_task(task_id=task_id)
    dataset = datasets.get_dataset(dataset_id=task.dataset_id)
    data, columns = dataset.get_data(return_attribute_names=True)
    train_indices, test_indices = task.get_train_test_split_indices()
    train = pd.DataFrame(data=data[train_indices], columns=columns)
    test = pd.DataFrame(data=data[test_indices], columns=columns)
    result_dir = openml_dir + '/task_id=' + str(task.dataset_id) + '/datasets/'

    if not os.path.exists(result_dir):
        try:
            os.makedirs(result_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    train.to_csv(result_dir + '/train.csv', index=False)
    test.to_csv(result_dir + '/test.csv', index=False)


def get_scikitlearn_flows(openml_dir, file_name='scikitlearn-flows.txt'):
    path = openml_dir + '/' + file_name
    if os.path.exists(path):
        with open(path, 'r') as in_file:
            return [int(flow_id) for flow_id in in_file.readlines()]
    flow_list = pd.DataFrame.from_dict(flows.list_flows(), orient='index')
    allowed = flow_list[flow_list.full_name.str.startswith('sklearn.pipeline')].id
    allowed = [f for f in allowed if f not in EXCLUDE_FLOWS]
    with open(openml_dir + '/' + file_name, 'w') as output:
        for a in allowed:
            output.write('{}\n'.format(a))
    return list(allowed)


def get_runs(openml_dir, flow_ids, task_id):
    path = openml_dir + '/task_id={}'.format(task_id) + '/all_runs.csv'
    if os.path.exists(path):
        return pd.read_csv(path, index_col=False)

    n = len(flow_ids) / 100
    run_dict = {}
    for i in range(n):
        run_dict.update(runs.list_runs(size=10000, task=[task_id], flow=flow_ids[(i * 100):(i + 1) * 100]))
    all_runs_pd = pd.DataFrame.from_dict(run_dict, orient='index')
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    all_runs_pd.to_csv(path, index=False)
    return all_runs_pd


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
    for v in flow.components.values():
        if v.class_name.startswith('helper.dual_imputer') or v.class_name.startswith('extra.dual_imputer'):
            print('undefined class: {}'.format(v.class_name))
            return 'ERROR'
        # if 'OneHotEncoder' in v.class_name:
        #     v.class_name = 'sklearn.preprocessing.OneHotEncoder'
        v.dependencies = u'sklearn>=0.19.1\nnumpy>=1.6.1\nscipy>=0.9'
        # for meta components (e.g, boosting algorithms)
        if hasattr(v, 'components'):
            for vv in v.components.values():
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
            print('skipping run {} because flow {} is not repairable'.format(run_id, run.flow_id))
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


def convert_to_tuple(openml_dir):
    flows_path = openml_dir + '/' + FLOW_DICTIONARY
    setups_path = openml_dir + '/' + SETUP_DICTIONARY
    setups_flow_path = openml_dir + '/' + SETUP_FLOW_LIST
    setup_flow = []
    with open(flows_path, 'rb') as g_input:
        flow_dict = pickle.load(g_input)

    with open(setups_path, 'rb') as g_input:
        setup_dict = pickle.load(g_input)

    for k, v in setup_dict.items():
        pipeline = flows.flow_to_sklearn(flow_dict[v.flow_id])
        setup_flow.append((v, pipeline))

    with open(setups_flow_path, 'wb') as output:
        pickle.dump(setup_flow, output, pickle.HIGHEST_PROTOCOL)


def get_filtered_setup_from_file(openml_dir):
    setups_flow_path = openml_dir + '/' + SETUP_FLOW_LIST
    with open(setups_flow_path, 'rb') as g_input:
        setup_flow = pickle.load(g_input)
    filtered_setups = pd.read_csv(openml_dir + '/' + FILTERED_SETUP, names=['setup_id'])
    return [(s, p) for s, p in setup_flow if s.setup_id in filtered_setups.setup_id.values]


def get_setup_from_file(openml_dir, limit):
    setups_flow_path = openml_dir + '/' + SETUP_FLOW_LIST
    with open(setups_flow_path, 'rb') as g_input:
        setup_flow = pickle.load(g_input)

    return setup_flow[:limit]
    # return setup_flow


def get_setup_and_pipeline(openml_dir, runs_file, limit=1000):
    flows_path = openml_dir + '/' + FLOW_DICTIONARY
    setups_path = openml_dir + '/' + SETUP_DICTIONARY

    if os.path.exists(flows_path):
        with open(flows_path, 'rb') as g_input:
            flow_dict = pickle.load(g_input)
    else:
        flow_dict = {}
    if os.path.exists(setups_path):
        with open(setups_path, 'rb') as g_input:
            setup_dict = pickle.load(g_input)
    else:
        setup_dict = {}
    runs_df = pd.read_csv(runs_file, index_col=False)[0:limit]
    setup_flow = []
    for index, row in runs_df.iterrows():
        if row['flow_id'] in flow_dict:
            flow = flow_dict[row['flow_id']]
        else:
            flow = repair(flows.get_flow(row['flow_id']))
            flow_dict[row['flow_id']] = flow

        if flow == 'ERROR':
            print('flow {} is not repairable'.format(row['flow_id']))
        else:
            try:
                pipeline = flows.flow_to_sklearn(flow)
                if row['setup_id'] in setup_dict:
                    setup = setup_dict[row['setup_id']]
                else:
                    setup = setups.get_setup(row['setup_id'])
                    setup_dict[row['setup_id']] = setup
                setup_flow.append((setup, pipeline))
            except Exception as err:
                print('Error while conversion to setup: {}'.format(err))

    with open(flows_path, 'wb') as output:
        pickle.dump(flow_dict, output, pickle.HIGHEST_PROTOCOL)

    with open(setups_path, 'wb') as output:
        pickle.dump(setup_dict, output, pickle.HIGHEST_PROTOCOL)

    return setup_flow


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        SOURCE_CODE_ROOT = sys.argv[1]
    else:
        SOURCE_CODE_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative' \
                           '-optimizer/ '
    sys.path.append(SOURCE_CODE_ROOT)
    from paper.experiment_helper import Parser

    parser = Parser(sys.argv)
    DEFAULT_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
    ROOT = parser.get('root', DEFAULT_ROOT)
    openml_task = int(parser.get('task', 31))
    ROOT_DATA_DIRECTORY = ROOT + '/data'
    openml_dir = ROOT_DATA_DIRECTORY + '/openml'
    config.set_cache_directory(openml_dir + '/cache')
    download_dataset(openml_dir=openml_dir, task_id=openml_task)
    flow_ids = get_scikitlearn_flows(openml_dir=openml_dir)
    get_runs(openml_dir=openml_dir, flow_ids=flow_ids, task_id=openml_task)
