#!/usr/bin/env python

"""Execution Time Experiment

Run a list of workloads in sequence and report the execution time for each one

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
from paper.experiments.scenario import get_kaggle_baseline_scenario, get_kaggle_optimized_scenario
from experiment_graph.executor import CollaborativeExecutor, BaselineExecutor
from experiment_graph.data_storage import DedupedStorageManager
from experiment_graph.optimizations.Reuse import FastBottomUpReuse
from paper.experiment_helper import Parser

parser = Parser(sys.argv)
verbose = parser.get('verbose', 0)
ROOT = parser.get('root', SOURCE_CODE_ROOT)

# Experiment Graph
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer

EXPERIMENT = parser.get('experiment', 'kaggle_home_credit')
ROOT_DATA_DIRECTORY = ROOT + '/data'

MODE = parser.get('mode', 'local')

EXPERIMENT_TIMESTAMP = datetime.now()

mat_budget = float(parser.get('mat_budget', '1.0')) * 1024.0 * 1024.0
method = parser.get('method', 'optimized')

# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]

rep = int(parser.get('rep', 2))

result_file = parser.get('result')


def run(executor, workload):
    if method == 'optimized':
        return executor.run_workload(workload=workload, root_data=ROOT_DATA_DIRECTORY, verbose=0)
    elif method == 'baseline':
        return executor.end_to_end_run(workload=workload, root_data=ROOT_DATA_DIRECTORY)


if not os.path.exists(os.path.dirname(result_file)):
    try:
        os.makedirs(os.path.dirname(result_file))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

if method == 'optimized':
    ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=FastBottomUpReuse.NAME)
    sa_materializer = StorageAwareMaterializer(storage_budget=mat_budget)
    executor = CollaborativeExecutor(ee, sa_materializer)
    workloads = get_kaggle_optimized_scenario()
elif method == 'baseline':
    executor = BaselineExecutor()
    workloads = get_kaggle_baseline_scenario()
else:
    raise Exception('invalid method: {}'.format(method))

for workload in workloads:
    workload_name = workload.__class__.__name__
    start = datetime.now()
    print '{}-Start of {} {} workload execution'.format(start, method, workload_name)
    success = run(executor, workload)
    end = datetime.now()
    print '{}-End of {} {} workload execution'.format(end, method, workload_name)

    elapsed = (end - start).total_seconds()

    if not success:
        elapsed = 'Failed!'

    with open(result_file, 'a') as the_file:
        # get_benchmark_results has the following order:
        the_file.write(
            '{},{},{},{},{},{},{}\n'.format(EXPERIMENT_TIMESTAMP.strftime("%H:%M:%S"), e_id,
                                            EXPERIMENT, workload_name, method, mat_budget, elapsed))

    if method == 'optimized':
        executor.local_process()
        executor.global_process()
        executor.cleanup()