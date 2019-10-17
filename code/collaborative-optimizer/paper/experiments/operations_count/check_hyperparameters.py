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
from experiment_graph.openml_helper.openml_connectors import *

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
from experiment_graph.executor import CollaborativeExecutor, BaselineExecutor
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer
from experiment_graph.optimizations.Reuse import LinearTimeReuse, AllComputeReuse
from experiment_graph.storage_managers import storage_profiler
from experiment_graph.openml_helper.openml_connectors import get_setup_and_pipeline
from experiment_graph.workloads.openml_optimized import OpenMLOptimizedWorkload
from experiment_graph.workloads.openml_baseline import OpenMLBaselineWorkload

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
limit = int(parser.get('limit', 2000))
openml_task = int(parser.get('task', 31))
OPENML_DIR = ROOT_DATA_DIRECTORY + '/openml/'
config.set_cache_directory(OPENML_DIR + '/cache')

result_file = parser.get('result', ROOT + '/experiment_results/local/operation_counts/openml/test.csv')
profile = storage_profiler.get_profile(parser.get('profile', ROOT_DATA_DIRECTORY + '/profiles/local-dedup'))

if not os.path.exists(os.path.dirname(result_file)):
    try:
        os.makedirs(os.path.dirname(result_file))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

method = parser.get('method', 'optimized')
should_warmstart = bool(int(parser.get('warmstart', 1)))

print 'running experiment {} with warmstarting: {}'.format(method, should_warmstart)

OPENML_DIR = ROOT_DATA_DIRECTORY + '/openml/'
OPENML_TASK = ROOT_DATA_DIRECTORY + '/openml/task_id={}'.format(openml_task)
setup_and_pipelines = get_setup_and_pipeline(openml_dir=OPENML_DIR, runs_file=OPENML_TASK + '/all_runs.csv',
                                             limit=limit)

i = 0
models = []
model_types = {}
warmstartable = {}
repetitions = 0
for setup, pipeline in setup_and_pipelines:
    edges = skpipeline_to_edge_list(pipeline=pipeline, setup=setup)

    model = edges[-1]

    i += 1
    n = model.__class__.__name__
    if n in model_types:
        model_types[n] += 1
    else:
        warmstartable[n] = hasattr(model, 'warm_start')
        model_types[n] = 1

    if model.get_params() in models:
        repetitions += 1
        print 'found matching model'
    models.append(model.get_params())

print '{} out of {} are repeated'.format(repetitions, i)
print model_types
print warmstartable
