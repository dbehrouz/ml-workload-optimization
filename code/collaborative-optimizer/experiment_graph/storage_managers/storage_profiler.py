import errno
import os
import sys

if len(sys.argv) > 1:
    SOURCE_CODE_ROOT = sys.argv[1]
else:
    SOURCE_CODE_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative' \
                       '-optimizer/ '

sys.path.append(SOURCE_CODE_ROOT)
from experiment_graph.graph.graph_representations import ExperimentGraph
from datetime import datetime
import json


def get_profile(profile_location):
    print open(profile_location, 'rb').read()
    return json.loads(open(profile_location, 'rb').read())


def profile_experiment_graph(profile_name, experiment_graph, result_folder, TRIAL=10):
    """
    :param profile_name:
    :type experiment_graph: ExperimentGraph
    :param result_folder:
    """

    time_size = {}
    graph = experiment_graph.graph
    for node, data in graph.nodes(data=True):
        if data['mat']:
            d_type = data['type']
            if d_type != 'SuperNode' and d_type != 'GroupBy':
                size = data['size']
                start = datetime.now()
                for i in range(TRIAL):
                    temp = experiment_graph.retrieve_data(node)
                end = datetime.now()
                time = ((end - start).microseconds / 1000.0) / TRIAL
                if d_type not in time_size:
                    time_size[d_type] = {'size': [], 'time': []}

                    time_size[d_type]['size'].append(size)
                    time_size[d_type]['time'].append(time)

    profile = {}
    for k, v in time_size.iteritems():
        profile[k] = sum(v['time']) / sum(v['size'])

    if not os.path.exists(result_folder):
        try:
            os.makedirs(result_folder)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    json.dump(profile, open(result_folder + '/' + profile_name, 'w'))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        SOURCE_CODE_ROOT = sys.argv[1]
    else:
        SOURCE_CODE_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative' \
                           '-optimizer/ '
    from paper.experiment_helper import Parser

    sys.path.append(SOURCE_CODE_ROOT)
    parser = Parser(sys.argv)
    trial = int(parser.get('trial', '5'))
    database_path = parser.get('experiment_graph',
                               '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/data'
                               '/experiment_graphs/kaggle_home_credit/start_here_a_gentle_introduction/all')
    result_folder = parser.get('result_folder',
                               '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/data/profiles')
    profile_name = parser.get('profile', 'local-dedup')
    from experiment_graph.execution_environment import ExecutionEnvironment

    ee = ExecutionEnvironment()
    ee.load_history_from_disk(database_path)

    profile_experiment_graph(profile_name, ee.experiment_graph, result_folder)
