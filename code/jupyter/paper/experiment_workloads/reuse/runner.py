#!/usr/bin/env python

"""Reuse Experiments Runner script
Executes multiple scripts (currently 3) after each other.
For optimized execution, the first script is running in normal mode as no experiment graph is constructed yet.
The rest of the scripts will use the experiment graph constructed so far to for operation and model reuse.
"""
import os
import uuid
from datetime import datetime

ROOT_PACKAGE_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter'
ROOT_DATA_DIRECTORY = ROOT_PACKAGE_DIRECTORY + '/data'
GRAPH_DATABASE_PATH = ROOT_PACKAGE_DIRECTORY + '/data/graph/exp-reuse.graph'

OUTPUT_CSV = 'results/run_times_same_workload.csv'
RESULT_FOLDER = 'results'
EXPERIMENT = 'kaggle-home-credit'
# TODO: This should change to three different scripts
WORKLOADS = ['workload_1', 'workload_1', 'workload_1']

# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]

for i in range(len(WORKLOADS)):
    run_number = i+1
    print 'Run Number {}'.format(run_number)
    # Running Optimized Workload 1 and storing the run time
    start = datetime.now()
    print '{}-Start of the Optimized Workload'.format(start)
    os.system(
        "python {}/optimized/{}.py {} {} {} {} {} {}".format(EXPERIMENT,
                                                             WORKLOADS[i],
                                                             ROOT_PACKAGE_DIRECTORY,
                                                             ROOT_DATA_DIRECTORY,
                                                             GRAPH_DATABASE_PATH,
                                                             # logs root directory
                                                             RESULT_FOLDER,
                                                             # experiment id for logging
                                                             e_id,
                                                             # run id for logging
                                                             run_number))
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print '{}-End of Optimized Workload'.format(end)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write('{},{},{},optimized,{},{}\n'.format(e_id, run_number, EXPERIMENT, WORKLOADS[i], elapsed))
    # End of Optimized Workload 1

    # # Running Baseline Workload 1 and storing the run time
    start = datetime.now()
    print '{}-Start of the Baseline Workload'.format(start)
    os.system("python {}/baseline/workload_1.py {}".format(EXPERIMENT, ROOT_DATA_DIRECTORY))
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print '{}-End of Baseline Workload'.format(end)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write(
            '{},{},{},baseline,{},{}\n'.format(e_id, run_number, EXPERIMENT, WORKLOADS[i], elapsed))
    # End of Baseline Workload 1
