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

from benchmark_helper import BenchmarkMetrics

ROOT_PACKAGE_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter'
sys.path.append(ROOT_PACKAGE_DIRECTORY)
# Experiment Graph
from experiment_graph.execution_environment import ExecutionEnvironment

EXPERIMENT = 'kaggle_home_credit'
WORKLOAD = 'introduction_to_manual_feature_engineering_p2'
ROOT_DATA_DIRECTORY = ROOT_PACKAGE_DIRECTORY + '/data'
DATABASE_PATH = ROOT_DATA_DIRECTORY + '/experiment_graphs/{}/environment_same_workload'.format(EXPERIMENT)


RESULT_FOLDER = ROOT_PACKAGE_DIRECTORY + '/reuse/same-workload/{}'.format(EXPERIMENT)
OUTPUT_CSV = RESULT_FOLDER + 'experiment_results.csv'
REP = 3

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

for i in range(1, REP + 1):
    print 'Run Number {}'.format(i)
    # Running Optimized Workload 1 and storing the run time
    execution_start = datetime.now()
    print '{}-Start of the Optimized Workload'.format(execution_start)
    optimized_workload = import_module(EXPERIMENT + '.optimized.' + WORKLOAD)
    ee.new_workload()
    optimized_workload.run(ee, ROOT_DATA_DIRECTORY)
    ee.update_history()
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()
    print '{}-End of Optimized Workload'.format(execution_end)

    with open(OUTPUT_CSV, 'a') as the_file:
        # get_benchmark_results has the following order:
        the_file.write('{},{},{},{},optimized,{}\n'.format(e_id, i, EXPERIMENT, WORKLOAD, elapsed))

    with open(RESULT_FOLDER + '/details/{}.csv'.format(e_id), 'a') as result:
        result.write(ee.get_benchmark_results() + "\n")

    # # Running Baseline Workload 1 and storing the run time
    execution_start = datetime.now()
    print '{}-Start of the Baseline Workload'.format(execution_start)
    baseline_workload = import_module(EXPERIMENT + '.baseline.' + WORKLOAD)
    baseline_workload.run(ROOT_DATA_DIRECTORY)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()
    print '{}-End of Baseline Workload in {} seconds'.format(execution_end, elapsed)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write(
            '{},{},{},{},baseline,{}\n'.format(e_id, i, EXPERIMENT, WORKLOAD, elapsed))
    # End of Baseline Workload 1
# ee.save_history(DATABASE_PATH)
