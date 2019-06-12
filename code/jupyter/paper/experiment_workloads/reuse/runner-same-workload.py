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

ROOT_PACKAGE_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter'
sys.path.append(ROOT_PACKAGE_DIRECTORY)
# Experiment Graph
from experiment_graph.execution_environment import ExecutionEnvironment

ROOT_DATA_DIRECTORY = ROOT_PACKAGE_DIRECTORY + '/data'
DATABASE_PATH = ROOT_PACKAGE_DIRECTORY + '/data/environment_different_workload'

OUTPUT_CSV = 'results/run_times_same_workload.csv'
RESULT_FOLDER = 'results'
EXPERIMENT = 'kaggle_home_credit'
REP = 2
WORKLOAD = 'introduction_to_manual_feature_engineering'

# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]

for i in range(1, REP + 1):
    print 'Run Number {}'.format(i)
    ee = ExecutionEnvironment()

    if os.path.isdir(DATABASE_PATH):
        print 'Load Existing Experiment Graph!!'
        execution_start = datetime.now()
        ee.load_history(DATABASE_PATH)
        load_time = (datetime.now() - execution_start).total_seconds()
    else:
        load_time = 0
        print 'No Experiment Graph Exists!!!'
    # Running Optimized Workload 1 and storing the run time
    execution_start = datetime.now()
    print '{}-Start of the Optimized Workload'.format(execution_start)
    optimized_workload = import_module(EXPERIMENT + '.optimized.' + WORKLOAD)
    model_training_time = optimized_workload.run(ee, ROOT_DATA_DIRECTORY)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    start = datetime.now()
    # Save the Graph to Disk
    # TODO: Maybe we need some versioning mechanism later on
    ee.save_history(environment_folder=DATABASE_PATH, overwrite=True)
    save_time = (datetime.now() - start).total_seconds()

    print '{}-End of Optimized Workload'.format(execution_end)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write(
            '{},{},{},{},optimized,{},{},{},{}\n'.format(e_id, i, EXPERIMENT, WORKLOAD,
                                                         elapsed, model_training_time, load_time, save_time))
    # End of Optimized Workload 1

    # # Running Baseline Workload 1 and storing the run time
    execution_start = datetime.now()
    print '{}-Start of the Baseline Workload'.format(execution_start)
    baseline_workload = import_module(EXPERIMENT + '.baseline.' + WORKLOAD)
    baseline_workload.run(ROOT_DATA_DIRECTORY)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()
    print '{}-End of Baseline Workload'.format(execution_end)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write(
            '{},{},{},{},baseline,{}\n'.format(e_id, i, EXPERIMENT, WORKLOAD, elapsed))
    # End of Baseline Workload 1
