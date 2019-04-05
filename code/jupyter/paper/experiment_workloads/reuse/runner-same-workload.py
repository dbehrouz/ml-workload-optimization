#!/usr/bin/env python

"""Reuse Experiments Runner script

Run the same workloads 2 times. The first time no experiment graph exists so both baseline and optimized
version will be long. The second run should be faster for optimized since the graph is
populated.

TODO: Currently the load and save time of the graph are also reported in the result

"""
import os
import uuid
from datetime import datetime

ROOT_PACKAGE_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter'
ROOT_DATA_DIRECTORY = ROOT_PACKAGE_DIRECTORY + '/data'
DATABASE_PATH = ROOT_PACKAGE_DIRECTORY + '/data/environment'

OUTPUT_CSV = 'results/run_times_same_workload.csv'
RESULT_FOLDER = 'results'
EXPERIMENT = 'kaggle-home-credit'
REP = 2
WORKLOAD = 'workload_1'

# unique identifier for the experiment run
e_id = uuid.uuid4().hex.upper()[0:8]

for i in range(1, REP + 1):
    print 'Run Number {}'.format(i)
    # Running Optimized Workload 1 and storing the run time
    start = datetime.now()
    print '{}-Start of the Optimized Workload'.format(start)
    os.system(
        "python {}/optimized/{}.py {} {} {} {} {} {}".format(EXPERIMENT,
                                                             WORKLOAD,
                                                             ROOT_PACKAGE_DIRECTORY,
                                                             ROOT_DATA_DIRECTORY,
                                                             DATABASE_PATH,
                                                             # logs root directory
                                                             RESULT_FOLDER,
                                                             # experiment id for logging
                                                             e_id,
                                                             # run id for logging
                                                             i))
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print '{}-End of Optimized Workload'.format(end)

    with open(OUTPUT_CSV, 'a') as the_file:
        the_file.write('{},{},{},optimized,{},{}\n'.format(e_id, i, EXPERIMENT, WORKLOAD, elapsed))
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
            '{},{},{},baseline,{},{}\n'.format(e_id, i, EXPERIMENT, WORKLOAD, elapsed))
    # End of Baseline Workload 1
