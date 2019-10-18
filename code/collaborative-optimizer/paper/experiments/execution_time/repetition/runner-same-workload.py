#!/usr/bin/env python

"""Repetition Experiments Runner script

Run the same workloads 2 times. The first time no experiment experiment_graphs exists so both baseline and optimized
version will be long. The second run should be faster for optimized since the experiment_graphs is
populated.

"""
import errno
import os
import sys
import uuid
from datetime import datetime



if len(sys.argv) > 1:
    SOURCE_CODE_ROOT = sys.argv[1]
else:
    SOURCE_CODE_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/'

sys.path.append(SOURCE_CODE_ROOT)
from paper.experiment_helper import ExperimentWorkloadFactory
from experiment_graph.executor import CollaborativeExecutor, BaselineExecutor
from experiment_graph.data_storage import DedupedStorageManager
from paper.experiment_helper import Parser
from experiment_graph.optimizations.Reuse import LinearTimeReuse
from experiment_graph.storage_managers import storage_profiler

parser = Parser(sys.argv)
verbose = parser.get('verbose', 0)
ROOT = parser.get('root', SOURCE_CODE_ROOT)

# Experiment Graph
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer

EXPERIMENT = parser.get('experiment', 'kaggle_home_credit')
WORKLOAD = parser.get('workload', 'introduction_to_manual_feature_engineering')
ROOT_DATA_DIRECTORY = ROOT + '/data'

MODE = parser.get('mode', 'local')

EXPERIMENT_TIMESTAMP = datetime.now()

mat_budget = float(parser.get('mat_budget', '1.0')) * 1024.0 * 1024.0
method = parser.get('method', 'optimized')

# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]

rep = int(parser.get('rep', 2))

result_file = parser.get('result')
profile = storage_profiler.get_profile(parser.get('profile', ROOT_DATA_DIRECTORY + '/profiles/local-dedup'))


def run(executor):
    if method == 'optimized':
        workload = ExperimentWorkloadFactory.get_workload(EXPERIMENT, method, WORKLOAD)
        return executor.run_workload(workload=workload, root_data=ROOT_DATA_DIRECTORY, verbose=0)
    elif method == 'baseline':
        workload = ExperimentWorkloadFactory.get_workload(EXPERIMENT, method, WORKLOAD)
        return executor.end_to_end_run(workload=workload, root_data=ROOT_DATA_DIRECTORY)


if not os.path.exists(os.path.dirname(result_file)):
    try:
        os.makedirs(os.path.dirname(result_file))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

if method == 'optimized':
    ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=LinearTimeReuse.NAME)
    sa_materializer = StorageAwareMaterializer(storage_budget=mat_budget)
    executor = CollaborativeExecutor(ee, cost_profile=profile, materializer=sa_materializer)
elif method == 'baseline':
    executor = BaselineExecutor()
else:
    raise Exception('invalid method: {}'.format(method))

i = 0
while i < rep:

    start = datetime.now()
    print '{}-Start of {} workload execution {}'.format(start, method, i + 1)
    success = run(executor)
    end = datetime.now()
    print '{}-End of {} workload execution {}'.format(end, method, i + 1)

    elapsed = (end - start).total_seconds()

    if not success:
        elapsed = 'Failed!'

    with open(result_file, 'a') as the_file:
        # get_benchmark_results has the following order:
        the_file.write(
            '{},{},{},{},{},{},{},{}\n'.format(EXPERIMENT_TIMESTAMP.strftime("%H:%M:%S"), e_id,
                                               EXPERIMENT, WORKLOAD, method, i + 1, mat_budget, elapsed))

    if method == 'optimized':
        executor.local_process()
        executor.global_process()
        executor.cleanup()

    i += 1
