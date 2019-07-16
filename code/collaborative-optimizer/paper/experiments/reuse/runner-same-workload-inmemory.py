#!/usr/bin/env python

"""Reuse Experiments Runner script

Run the same workloads 2 times. The first time no experiment experiment_graphs exists so both baseline and optimized
version will be long. The second run should be faster for optimized since the experiment_graphs is
populated.

TODO: Currently the load and save time of the experiment_graphs are also reported in the result

"""
import os
import sys
import uuid
from datetime import datetime
from importlib import import_module

if len(sys.argv) > 1:
    SOURCE_CODE_ROOT = sys.argv[1]
else:
    SOURCE_CODE_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/'

sys.path.append(SOURCE_CODE_ROOT)
from experiment_graph.benchmark_helper import BenchmarkMetrics
from paper.experiment_helper import Parser

parser = Parser(sys.argv)
verbose = parser.get('verbose', 0)
ROOT = parser.get('root', '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/')
# Experiment Graph
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer

EXPERIMENT = parser.get('experiment', 'kaggle_home_credit')
WORKLOAD = parser.get('workload', 'introduction_to_manual_feature_engineering')
ROOT_DATA_DIRECTORY = ROOT + '/data'

MODE = parser.get('mode', 'local')
RESULT_FOLDER = ROOT + '/experiment_results/{}/reuse/same-workload/{}'.format(MODE, EXPERIMENT)

OUTPUT_CSV = RESULT_FOLDER + '/experiment_results.csv'

# what percentage of the total artifact size to materialize
mat_rate = float(parser.get('mat_rate', 0.0))
run_baseline = parser.get('run_baseline', 'no')

# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]
ee = ExecutionEnvironment()


def run(experiment_env):
    experiment_env.new_workload()
    optimized_workload = import_module('paper.experiment_workloads.' + EXPERIMENT + '.optimized.' + WORKLOAD)
    optimized_workload.run(experiment_env, ROOT_DATA_DIRECTORY, verbose=verbose)
    experiment_env.update_history()


if not os.path.isdir(RESULT_FOLDER + '/details/'):
    os.makedirs(RESULT_FOLDER + '/details/')

with open(RESULT_FOLDER + '/details/{}.csv'.format(e_id), 'w') as result:
    result.write(','.join(BenchmarkMetrics.keys) + "\n")

EXPERIMENT_TIMESTAMP = datetime.now()

if run_baseline == 'no':
    ee = ExecutionEnvironment()
    default_start = datetime.now()
    print '{}-Start of the Default Optimized Workload'.format(default_start)
    run(ee)
    default_end = datetime.now()
    print '{}-End of First Optimized Workload'.format(default_end)

    if 0.0 < mat_rate:
        total_size = ee.history_graph.get_total_size()
        budget = mat_rate * total_size
        print 'total graph size: {}, data storage size: {}, real size: {}'.format(
            total_size, ee.data_storage.total_size(),
            ee.get_real_history_graph_size())
        print 'materialization rate: {}, budget: {}'.format(mat_rate, budget)
        sa_materializer = StorageAwareMaterializer(execution_environment=ee, storage_budget=budget,
                                                   verbose=False)
        sa_materializer.run_and_materialize()
        print 'after materialization graph size: {}, data storage size: {}, real size: {}'.format(
            ee.history_graph.get_total_materialized_size(), ee.data_storage.total_size(),
            ee.get_real_history_graph_size())
        real_start = datetime.now()
        run(ee)
        real_end = datetime.now()
        elapsed = (real_end - real_start).total_seconds()
    elif mat_rate == 0.0:
        elapsed = (default_end - default_start).total_seconds()
    else:
        raise Exception('Invalid materialization rate: {}'.format(mat_rate))

    with open(OUTPUT_CSV, 'a') as the_file:
        # get_benchmark_results has the following order:
        the_file.write(
            '{},{},{},{},optimized,{},{}\n'.format(EXPERIMENT_TIMESTAMP.strftime("%Y-%m-%d %H:%M:%S"), e_id, EXPERIMENT,
                                                   WORKLOAD, mat_rate, elapsed))

    with open(RESULT_FOLDER + '/details/{}.csv'.format(e_id), 'a') as result:
        result.write(ee.get_benchmark_results() + "\n")
    # Code for materialization
    del ee

else:
    # Running Baseline Workload and storing the run time
    default_start = datetime.now()
    print '{}-Start of the Baseline Workload'.format(default_start)
    baseline_workload = import_module('paper.experiment_workloads.' + EXPERIMENT + '.baseline.' + WORKLOAD)
    baseline_workload.run(ROOT_DATA_DIRECTORY)
    default_end = datetime.now()
    elapsed = (default_end - default_start).total_seconds()
    print '{}-End of Baseline Workload in {} seconds'.format(default_end, elapsed)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write(
            '{},{},{},{},baseline,{}\n'.format(EXPERIMENT_TIMESTAMP.strftime("%Y-%m-%d %H:%M:%S"), e_id, EXPERIMENT,
                                               WORKLOAD, elapsed))
