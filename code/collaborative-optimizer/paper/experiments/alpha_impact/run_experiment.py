#!/usr/bin/env python

"""Execution Time Experiment

Run a list of workloads in sequence and report the execution time for each one
These are the flow ids with number of setups that end up being executed:
{5804: 18,
 5909: 9,
 5910: 1,
 5913: 1,
 5914: 1,
 5995: 1,
 6268: 3,
 6269: 1,
 6334: 1,
 6840: 341,
 6946: 2,
 6952: 31,
 6954: 7,
 6958: 1,
 6969: 1503,
 6970: 79,
 5804: 18
}
 Complete list of setups are the experiment result files

"""
import errno
import os
import sys
import uuid
from datetime import datetime

from openml import config

if len(sys.argv) > 1:
    SOURCE_CODE_ROOT = sys.argv[1]
else:
    SOURCE_CODE_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative' \
                       '-optimizer/ '
sys.path.append(SOURCE_CODE_ROOT)
# Somehow someone hard codes this to be on top of the sys path and I cannot get rid of it
if '/home/zeuchste/git/scikit-learn' in sys.path:
    sys.path.remove('/home/zeuchste/git/scikit-learn')

from paper.experiment_helper import Parser
from experiment_graph.data_storage import StorageManagerFactory, DedupedStorageManager
from experiment_graph.executor import CollaborativeExecutor
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer, \
    TopNModelMaterializer, OracleBestModelMaterializer
from experiment_graph.optimizations.Reuse import LinearTimeReuse
from experiment_graph.storage_managers import storage_profiler
from experiment_graph.openml_helper.openml_connectors import get_setup_and_pipeline
from experiment_graph.workloads.openml_optimized import OpenMLOptimizedWorkload

e_id = uuid.uuid4().hex.upper()[0:8]
EXPERIMENT_TIMESTAMP = datetime.now()

parser = Parser(sys.argv)
verbose = parser.get('verbose', 0)

DEFAULT_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
ROOT = parser.get('root', DEFAULT_ROOT)
ROOT_DATA_DIRECTORY = ROOT + '/data'

mat_budget = float(parser.get('mat_budget', '1.0')) * 1024.0 * 1024.0
materializer = StorageAwareMaterializer(storage_budget=mat_budget)

storage_manager = StorageManagerFactory.get_storage(parser.get('storage_type', 'dedup'))

EXPERIMENT = parser.get('experiment', 'openml')
limit = int(parser.get('limit', 100))
openml_task = int(parser.get('task', 31))
OPENML_DIR = ROOT_DATA_DIRECTORY + '/openml/'
config.set_cache_directory(OPENML_DIR + '/cache')

result_file = parser.get('result', ROOT + '/experiment_results/local/alpha_impact/openml/test.csv')
profile = storage_profiler.get_profile(parser.get('profile', ROOT_DATA_DIRECTORY + '/profiles/local-dedup'))

if not os.path.exists(os.path.dirname(result_file)):
    try:
        os.makedirs(os.path.dirname(result_file))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

method = parser.get('method', 'optimized')

OPENML_DIR = ROOT_DATA_DIRECTORY + '/openml/'
OPENML_TASK = ROOT_DATA_DIRECTORY + '/openml/task_id={}'.format(openml_task)
setup_and_pipelines = get_setup_and_pipeline(openml_dir=OPENML_DIR, runs_file=OPENML_TASK + '/all_runs.csv',
                                             limit=limit)

mat_type = parser.get('mat_type', 'best_n')
alpha = float(parser.get('alpha', '0.5'))

if mat_type == 'best_n':
    materializer = TopNModelMaterializer(n=1, alpha=alpha)
else:
    materializer = OracleBestModelMaterializer()
ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=LinearTimeReuse.NAME)
executor = CollaborativeExecutor(ee, cost_profile=profile, materializer=materializer)


def get_workload(setup, pipeline):
    return OpenMLOptimizedWorkload(setup, pipeline, task_id=openml_task)


def run(executor, workload):
    if method == 'optimized' or method == 'mock_optimized':
        return executor.run_workload(workload=workload, root_data=ROOT_DATA_DIRECTORY, verbose=verbose)
    elif method == 'baseline':
        return executor.end_to_end_run(workload=workload, root_data=ROOT_DATA_DIRECTORY)
    elif method == 'mock':
        return executor.end_to_end_run(workload=workload, root_data=ROOT_DATA_DIRECTORY)


def is_best_model_materialized(executor):
    graph = executor.execution_environment.experiment_graph.graph
    best = (0, {}, 'id')
    for n, d in graph.nodes(data=True):
        if d['type'] == 'SK_Model' and d['score'] > 0:
            if best[0] < d['score']:
                best = (d['score'], d, n)
    return graph.nodes[best[2]]['mat']


best_workload = None
best_score = -1
best_setup = -1
best_pipeline = -1
print 'experiment with materializer: {}, alpha: {}'.format(mat_type, alpha)
for setup, pipeline in setup_and_pipelines:

    workload = get_workload(setup, pipeline)
    start = datetime.now()
    # print '{}-Start of {} with pipeline {}, execution'.format(start, workload_name)
    success = run(executor, workload)
    end_current = datetime.now()
    run_time_current = (end_current - start).total_seconds()
    current_score = workload.get_score()
    if best_score == -1:
        best_score = current_score
        best_setup = setup.setup_id
        best_pipeline = setup.flow_id
    if best_workload is not None:
        start_best_workload = datetime.now()
        success = run(executor, best_workload)
        run_time_best = (datetime.now() - start_best_workload).total_seconds()
        if current_score > best_score:
            best_score = current_score
            best_workload = workload
            best_setup = setup.setup_id
            best_pipeline = setup.flow_id
    else:
        run_time_best = run_time_current
        best_workload = workload
        best_score = current_score
    end = datetime.now()

    elapsed = (end - start).total_seconds()

    executor.local_process()
    executor.global_process()
    executor.cleanup()
    mat_status = 1 if is_best_model_materialized(executor) else 0

    if not success:
        elapsed = 'Failed!'
    # graph = executor.execution_environment.experiment_graph
    # total_mat = graph.get_total_materialized_size()
    # total_size = graph.get_total_size()
    with open(result_file, 'a') as the_file:
        # get_benchmark_results has the following order:
        the_file.write(
            '{},{},{},{},{},{},{},{},{}\n'.format(EXPERIMENT_TIMESTAMP.strftime("%H:%M:%S"), e_id,
                                                  EXPERIMENT, setup.flow_id, setup.setup_id, mat_type,
                                                  alpha, mat_status,
                                                  elapsed))
