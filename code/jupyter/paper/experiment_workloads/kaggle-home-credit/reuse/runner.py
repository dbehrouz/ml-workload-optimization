#!/usr/bin/env python

"""Reuse Experiments Runner script

"""
import os
from datetime import datetime

ROOT_DATA_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter/data'
ROOT_PACKAGE_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter'
for i in range(1, 4):
    print 'Run Number {}'.format(i)
    # Running Optimized Workload 1 and storing the run time
    start_time_optimized = datetime.now()
    print '{}-Start of the Optimized Workload'.format(start_time_optimized)
    os.system(
        "python optimized/workload_1.py {} {}".format(ROOT_PACKAGE_DIRECTORY, ROOT_DATA_DIRECTORY))
    end_time_optimized = datetime.now()
    print '{}-End of Optimized Workload'.format(end_time_optimized)

    with open('run_times.csv', 'a') as the_file:
        the_file.write('{},kaggle-home-credit,optimized,workload_1,{}\n'.format(i, (
                end_time_optimized - start_time_optimized).total_seconds()))
    # End of Optimized Workload 1

    # Running Baseline Workload 1 and storing the run time
    start_time_baseline = datetime.now()
    print '{}-Start of the Baseline Workload'.format(start_time_baseline)
    os.system("python baseline/workload_1.py {}".format(ROOT_DATA_DIRECTORY))
    end_time_baseline = datetime.now()
    print '{}-End of Baseline Workload'.format(end_time_baseline)

    with open('run_times.csv', 'a') as the_file:
        the_file.write(
            '{},kaggle-home-credit,baseline,workload_1,{}\n'.format(i, (
                    end_time_baseline - start_time_baseline).total_seconds()))
    # End of Baseline Workload 1
