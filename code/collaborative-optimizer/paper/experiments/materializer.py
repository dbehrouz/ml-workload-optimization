"""
This script receives the name of a workload and a materialization rate (or an array of rates).
It then executes the workload and materializes the graph based on the given rates and writes the
execution environment object to file
"""
import copy
import os
import sys
from importlib import import_module

if len(sys.argv) > 1:
    SOURCE_CODE_ROOT = sys.argv[1]
else:
    SOURCE_CODE_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/'

sys.path.append(SOURCE_CODE_ROOT)
from paper.experiment_helper import Parser

parser = Parser(sys.argv)
verbose = int(parser.get('verbose', 0))
ROOT = parser.get('root', '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/')
# Experiment Graph
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer

EXPERIMENT = parser.get('experiment', 'kaggle_home_credit')
WORKLOAD = parser.get('workload', 'introduction_to_manual_feature_engineering')
ROOT_DATA_DIRECTORY = ROOT + '/data'
DATABASE_PATH = ROOT_DATA_DIRECTORY + '/experiment_graphs/{}/{}'.format(EXPERIMENT, WORKLOAD)

MODE = parser.get('mode', 'local')

MAT_RATES = [float(m) for m in parser.get('mat_rates', '1.0').split(',')]

ee = ExecutionEnvironment()
ee.new_workload()
optimized_workload = import_module('paper.experiment_workloads.' + EXPERIMENT + '.optimized.' + WORKLOAD)
optimized_workload.run(ee, ROOT_DATA_DIRECTORY, verbose=verbose)
ee.update_history()

for rate in MAT_RATES:
    print 'materialization for rate: {}'.format(rate)
    ee_storage_aware = copy.deepcopy(ee)

    total_size = ee_storage_aware.experiment_graph.get_total_size()
    budget = rate * total_size
    if 0.0 < rate:
        print 'total graph size: {}, data storage size: {}, real size: {}'.format(
            total_size, ee_storage_aware.data_storage.total_size(),
            ee_storage_aware.get_real_history_graph_size())
        print 'materialization rate: {}, budget: {}'.format(rate, budget)
        sa_materializer = StorageAwareMaterializer(execution_environment=ee_storage_aware, storage_budget=budget,
                                                   verbose=False)

        sa_materializer.run_and_materialize()
        print 'after materialization graph size: {}, data storage size: {}, real size: {}'.format(
            ee_storage_aware.experiment_graph.get_total_materialized_size(), ee_storage_aware.data_storage.total_size(),
            ee_storage_aware.get_real_history_graph_size())
    if rate != 0.0:
        if not os.path.isdir(DATABASE_PATH):
            os.makedirs(DATABASE_PATH)
        ee_storage_aware.save_history(DATABASE_PATH + '/mat_rate_{}'.format(rate))

    del ee_storage_aware
