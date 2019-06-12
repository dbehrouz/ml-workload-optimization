#!/usr/bin/env python

"""Reuse Experiments Runner script
Executes multiple scripts (currently 3) after each other.
For optimized execution, the first script is running in normal mode as no experiment experiment_graphs is constructed yet.
The rest of the scripts will use the experiment experiment_graphs constructed so far to for operation and model reuse.
"""
import os
import uuid
import sys
from datetime import datetime
from importlib import import_module

ROOT_PACKAGE_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter'
sys.path.append(ROOT_PACKAGE_DIRECTORY)
# Experiment Graph
from experiment_graph.execution_environment import ExecutionEnvironment

ROOT_DATA_DIRECTORY = ROOT_PACKAGE_DIRECTORY + '/data'
DATABASE_PATH = ROOT_PACKAGE_DIRECTORY + '/data/graphs/home-credit-default-risk/environment_different_workload'

OUTPUT_CSV = 'results/run_times_different_workload.csv'
RESULT_FOLDER = 'results'
EXPERIMENT = 'kaggle_home_credit'
# TODO: This should change to three different scripts
WORKLOADS = ['start_here_a_gentle_introduction', 'introduction_to_manual_feature_engineering']

# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]

for i in range(len(WORKLOADS)):
    print 'Run Number {}'.format(i + 1)
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
    optimized_workload = import_module(EXPERIMENT + '.optimized.' + WORKLOADS[i])
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
            '{},{},{},{},optimized,{},{},{},{}\n'.format(e_id, i, EXPERIMENT, WORKLOADS[i],
                                                         elapsed, model_training_time, load_time, save_time))
    # End of Optimized Workload 1

    # # Running Baseline Workload 1 and storing the run time
    execution_start = datetime.now()
    print '{}-Start of the Baseline Workload'.format(execution_start)
    baseline_workload = import_module(EXPERIMENT + '.baseline.' + WORKLOADS[i])
    baseline_workload.run(ROOT_DATA_DIRECTORY)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()
    print '{}-End of Baseline Workload'.format(execution_end)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write(
            '{},{},{},{},baseline,{}\n'.format(e_id, i, EXPERIMENT, WORKLOADS[i], elapsed))
    # End of Baseline Workload 1
