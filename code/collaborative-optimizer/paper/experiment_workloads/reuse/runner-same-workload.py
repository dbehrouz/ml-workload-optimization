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
verbose = int(parser.get('verbose', 0))
ROOT = parser.get('root', '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/')
# Experiment Graph

from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer

EXPERIMENT = parser.get('experiment', 'kaggle_home_credit')
WORKLOAD = parser.get('workload', 'introduction_to_manual_feature_engineering_p2')
ROOT_DATA_DIRECTORY = ROOT + '/data'
DATABASE_PATH = ROOT + '/experiment_graphs/{}/environment_same_workload'.format(EXPERIMENT)

MODE = parser.get('mode', 'local')
RESULT_FOLDER = ROOT + '/experiment_results/{}/reuse/same-workload/{}'.format(MODE, EXPERIMENT)
OUTPUT_CSV = RESULT_FOLDER + '/experiment_results.csv'
REP = int(parser.get('rep', 2))
# what percentage of the total artifact size to materialize
MATERIALIZATION_RATES = [float(m) for m in parser.get('mat_rates', '1.0').split(',')]
RUN_BASELINE = parser.get('run_baseline', 'no')

# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]
ee = ExecutionEnvironment()

if not os.path.isdir(RESULT_FOLDER + '/details/'):
    os.makedirs(RESULT_FOLDER + '/details/')

with open(RESULT_FOLDER + '/details/{}.csv'.format(e_id), 'w') as result:
    result.write(','.join(BenchmarkMetrics.keys) + "\n")

if os.path.isdir(DATABASE_PATH):
    print 'Load Existing Experiment Graph!!'
    ee.load_history_from_disk(DATABASE_PATH)
else:
    print 'No Experiment Graph Exists!!!'
for mat_rate in MATERIALIZATION_RATES:
    for i in range(1, REP + 1):
        print 'Run Number {}, for script {}, with materialization rate: {}'.format(i, WORKLOAD, mat_rate)
        # Running Optimized Workload 1 and storing the run time
        execution_start = datetime.now()
        print '{}-Start of the Optimized Workload'.format(execution_start)
        optimized_workload = import_module(EXPERIMENT + '.optimized.' + WORKLOAD)
        ee.new_workload()
        optimized_workload.run(ee, ROOT_DATA_DIRECTORY, verbose=verbose)
        ee.update_history()
        execution_end = datetime.now()
        elapsed = (execution_end - execution_start).total_seconds()
        print '{}-End of Optimized Workload'.format(execution_end)

        with open(OUTPUT_CSV, 'a') as the_file:
            # get_benchmark_results has the following order:
            the_file.write('{},{},{},{},{}, optimized,{}\n'.format(e_id, i, EXPERIMENT, WORKLOAD, mat_rate, elapsed))

        with open(RESULT_FOLDER + '/details/{}.csv'.format(e_id), 'a') as result:
            result.write(ee.get_benchmark_results() + "\n")
        # Code for materialization
        total_graph_size = ee.history_graph.get_total_size()

        if mat_rate != 1.0:
            budget = mat_rate * total_graph_size
            print 'total size {}, materialization rate: {}, budget: {}'.format(total_graph_size, mat_rate, budget)
            sa_materializer = StorageAwareMaterializer(execution_environment=ee, storage_budget=budget, verbose=False)
            sa_materializer.run_and_materialize()
            print 'after materialization graph size: {}, data storage size: {}, real size: {}'.format(
                ee.history_graph.get_total_materialized_size(), ee.data_storage.total_size(),
                ee.get_real_history_graph_size())
        elif mat_rate == 0.0:
            print 'materialization rate is 0.0'
            del ee
            ee = ExecutionEnvironment()
        else:
            print 'ignoring materialization, since materialization rate is 1.0'

for i in range(1, REP + 1):
    # Running Baseline Workload and storing the run time
    execution_start = datetime.now()
    print '{}-Start of the Baseline Workload'.format(execution_start)
    baseline_workload = import_module(EXPERIMENT + '.baseline.' + WORKLOAD)
    baseline_workload.run(ROOT_DATA_DIRECTORY)
    execution_end = datetime.now()
    elapsed = (execution_start - execution_start).total_seconds()
    print '{}-End of Baseline Workload in {} seconds'.format(execution_end, elapsed)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write(
            '{},{},{},{},baseline,{}\n'.format(e_id, i, EXPERIMENT, WORKLOAD, elapsed))

    # End of Baseline Workload 1
