#!/usr/bin/env python

"""Execution Time Experiment

Run a list of workloads in sequence and report the execution time for each one

"""
import errno
import os
import random
import sys

import pickle
import networkx as nx

# Somehow someone hard codes this to be on top of the sys path and I cannot get rid of it
if '/home/zeuchste/git/scikit-learn' in sys.path:
    sys.path.remove('/home/zeuchste/git/scikit-learn')

from experiment_graph.data_storage import DedupedStorageManager
from paper.experiment_helper import Parser
from experiment_graph.data_storage import StorageManagerFactory
from experiment_graph.graph.node import *
from experiment_graph.optimizations.Reuse import LinearTimeReuse
from experiment_graph.optimizations.Reuse import HelixReuse

parser = Parser(sys.argv)
verbose = parser.get('verbose', 0)
DEFAULT_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
ROOT = parser.get('root', DEFAULT_ROOT)

# Experiment Graph
from experiment_graph.execution_environment import ExecutionEnvironment

EXPERIMENT = parser.get('experiment', 'kaggle_home_credit')
ROOT_DATA_DIRECTORY = ROOT + '/data'

EXPERIMENT_TIMESTAMP = datetime.now()

reuse_type = parser.get('reuse_type', LinearTimeReuse.NAME)
max_load = parser.get('max_load', 50)
max_compute = parser.get('max_compute', 100)
mat_prob = parser.get('mat_prob', 0.05)
WORKLOAD_SIZE_MIN = int(parser.get('workload_size_min', 500))
WORKLOAD_SIZE_MAX = int(parser.get('workload_size_min', 1000))
NUMBER_OF_WORKLOADS = int(parser.get('number_of_workloads', 100))
ITERATION = int(parser.get('iter', 100))
REP = int(parser.get('rep', 3))
# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]

result_file = parser.get('result', ROOT + '/experiment_results/local/reuse_benchmarking/test.csv')

if not os.path.exists(os.path.dirname(result_file)):
    try:
        os.makedirs(os.path.dirname(result_file))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=reuse_type)
graph_path = parser.get('graph_path', ROOT_DATA_DIRECTORY + '/experiment_graphs/kaggle_home_credit/benchmarking/graph')
with open(graph_path, 'rb') as g_input:
    graph = pickle.load(g_input)

in_degree = pd.DataFrame([(n, d) for n, d in graph.in_degree()], columns=['node', 'in_degree'])
out_degree = pd.DataFrame([(n, d) for n, d in graph.out_degree()], columns=['node', 'out_degree'])
MAT_CHANCE = mat_prob
COMPUTE_COST_DIST = range(1, max_compute)
LOAD_COST_DIST = range(1, max_load)
GRAPH_STATS = in_degree.merge(out_degree, on='node')
WORKLOAD_SIZE_DIST = range(WORKLOAD_SIZE_MIN, WORKLOAD_SIZE_MAX)

SEED_VALUE = 42


def generate_nx_graph(graph_size):
    graph_obj = nx.DiGraph()
    graph_obj.add_node('ROOT',
                       **{'root': True, 'mat': True, 'type': 'Dataset', 'compute_cost': 2,
                          'load_cost': 1, 'data': Mock_object()})
    leaves = ['ROOT']
    for _ in range(graph_size):
        r = random.randint(0, len(leaves) - 1)
        current_leaf = leaves.pop(r)
        k = random.choice(GRAPH_STATS['out_degree'])
        if k == 0:
            k += 1
        for _ in range(k):
            compute_cost = random.choice(COMPUTE_COST_DIST)
            load_cost = random.choice(LOAD_COST_DIST)

            if load_cost < compute_cost:
                mat = random.uniform(0, 1) < MAT_CHANCE
            else:
                mat = False
            node_id = uuid.uuid4().hex.upper()[0:16]
            graph_obj.add_node(node_id, **{'type': 'Dataset', 'mat': mat, 'compute_cost': compute_cost,
                                           'load_cost': load_cost, 'data': Mock_object()})
            graph_obj.add_edge(current_leaf, node_id)
            leaves.append(node_id)

        k = random.choice(GRAPH_STATS['in_degree'])
        if 1 < k <= len(leaves):
            compute_cost = random.choice(COMPUTE_COST_DIST)
            load_cost = random.choice(LOAD_COST_DIST)
            if load_cost < compute_cost:
                mat = random.uniform(0, 1) < MAT_CHANCE
            else:
                mat = False
            node_id = uuid.uuid4().hex.upper()[0:16]
            graph_obj.add_node(node_id, **{'type': 'Dataset', 'mat': mat, 'compute_cost': compute_cost,
                                           'load_cost': load_cost, 'data': Mock_object()})

            for _ in range(k):
                r = random.randint(0, len(leaves) - 1)
                join_node = leaves.pop(r)
                graph_obj.add_edge(join_node, node_id)

            leaves.append(node_id)

    return graph_obj, len(graph_obj.nodes()), leaves


class Mock_object:
    def __init__(self):
        self.computed = False


print('Starting the Experiments with {} Workloads and {} Iterations'.format(NUMBER_OF_WORKLOADS, ITERATION))

all_workloads = []
for _ in range(NUMBER_OF_WORKLOADS):
    workload_size = random.choice(WORKLOAD_SIZE_DIST)
    all_workloads.append(generate_nx_graph(workload_size))

all_indices = [random.randint(0, len(all_workloads) - 1) for _ in range(ITERATION)]

for rep in range(1, REP + 1):
    storage_manager = StorageManagerFactory.get_storage('dedup')
    execution_environment = ExecutionEnvironment(storage_manager, reuse_type='linear')
    experiment_id = uuid.uuid4().hex.upper()[0:8]
    print('starting rep: {}'.format(rep))
    times = []
    i = 0
    for w in all_indices:
        i += 1
        execution_environment.workload_dag.graph, workload_size, leaves = all_workloads[w]

        vertex = random.choice(leaves)
        workload = execution_environment.workload_dag
        history = execution_environment.experiment_graph
        execution_environment.mock_update_history()

        start_linear = datetime.now()
        linear_mat, _, _ = LinearTimeReuse().run(vertex=vertex, workload=workload, history=history, verbose=0)
        total_linear = (datetime.now() - start_linear).total_seconds()

        start_helix = datetime.now()
        helix_mat, _, _ = HelixReuse().run(vertex=vertex, workload=workload, history=history, verbose=0)
        total_helix = (datetime.now() - start_helix).total_seconds()

        times.append((experiment_id, i, workload_size, total_linear, total_helix))
        execution_environment.new_workload()
        # assert linear_mat == helix_mat

    with open(result_file, 'a') as the_file:
        the_file.write('\n'.join('{},{},{},{},{}'.format(x[0], x[1], x[2], x[3], x[4]) for x in times))
        the_file.write('\n')
