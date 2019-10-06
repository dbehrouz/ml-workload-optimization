#!/usr/bin/env python

"""Optimized Fork of Joined Workload 1 and 2

This script is the optimized version of the workload 'fork_join_start_here_intro.py'
which utilizes our Experiment Graph for optimizing the workload
"""
import os
import warnings
# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib

from experiment_graph.workload import Workload

matplotlib.use('ps')

import matplotlib.pyplot as plt
import numpy as np
# numpy and pandas for data manipulation
import pandas as pd
import seaborn as sns

# Experiment Graph

# Suppress warnings
warnings.filterwarnings('ignore')


class fork_join_start_here_intro(Workload):

    def run(self, execution_environment, root_data, verbose=0):
        print(os.listdir(root_data))
        app_train = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        print('Training data shape: ', app_train.shape().data(verbose))
        app_train.head().data(verbose)

        app_test = execution_environment.load(root_data + '/kaggle_home_credit/application_test.csv')
        print('Testing data shape: ', app_test.shape().data(verbose))
        app_test.head().data(verbose)

        test_labels = execution_environment.load(root_data + '/kaggle_home_credit/application_test_labels.csv')

        app_train['TARGET'].value_counts().data(verbose)

        app_train['TARGET'].data(verbose).astype(int).plot.hist()

        # Function to calculate missing values by column# Funct
        def missing_values_table(dataset):
            # Total missing values
            mis_val = dataset.isnull().sum().data(verbose)

            mis_val_percent = 100 * mis_val / len(dataset.data(verbose))

            # Make a table with the results
            mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
            # Rename the columns
            mis_val_table_ren_columns = mis_val_table.rename(columns={
                0: 'Missing Values',
                1: '% of Total Values'
            })
            # Sort the table by percentage of missing descending
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
                '% of Total Values', ascending=False).round(1)

            # Print some summary information
            print("Your selected dataframe has " + str(dataset.shape().data(verbose)[1]) + " columns.\n"
                                                                                           "There are " + str(
                mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")

            # Return the dataframe with missing information
            return mis_val_table_ren_columns

        missing_values = missing_values_table(app_train)
        missing_values.head(20)

        app_train.dtypes().data(verbose).value_counts()

        app_train.select_dtypes('object').nunique().data(verbose)

        from experiment_graph.sklearn_helper.preprocessing import LabelEncoder
        # Create a label encoder object
        le_count = 0

        columns = app_train.select_dtypes('object').data(verbose).columns
        for col in columns:
            # we are not using nunique because it discard nan
            if app_train[col].nunique(dropna=False).data(verbose) <= 2:
                le = LabelEncoder()
                le.fit(app_train[col])

                app_train = app_train.replace_columns(col, le.transform(app_train[col]))
                app_test = app_test.replace_columns(col, le.transform(app_test[col]))

                # Keep track of how many columns were label encoded
                le_count += 1
        print('%d columns were label encoded.' % le_count)
        app_train.data(verbose)
        app_test.data(verbose)

        app_train = app_train.onehot_encode()
        app_test = app_test.onehot_encode()

        print('Training Features shape: ', app_train.shape().data(verbose))
        print('Testing Features shape: ', app_test.shape().data(verbose))

        train_labels = app_train['TARGET']
        train_columns = app_train.data(verbose).columns
        test_columns = app_test.data(verbose).columns
        for c in train_columns:
            if c not in test_columns:
                app_train = app_train.drop(c)

        app_train = app_train.add_columns('TARGET', train_labels)

        print('Training Features shape: ', app_train.shape().data(verbose))
        print('Testing Features shape: ', app_test.shape().data(verbose))

        (app_train['DAYS_BIRTH'] / 365).describe().data(verbose)

        app_train['DAYS_EMPLOYED'].describe().data(verbose)

        app_train['DAYS_EMPLOYED'].data(verbose).plot.hist(title='Days Employment Histogram')
        plt.xlabel('Days Employment')

        anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
        non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
        print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean().data(verbose)))
        print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean().data(verbose)))
        print('There are %d anomalous days of employment' % anom.shape().data(verbose)[0])

        days_employed_anom = app_train["DAYS_EMPLOYED"] == 365243
        app_train = app_train.add_columns('DAYS_EMPLOYED_ANOM', days_employed_anom)
        temp = app_train['DAYS_EMPLOYED'].replace({365243: np.nan})
        app_train = app_train.drop('DAYS_EMPLOYED')
        app_train = app_train.add_columns('DAYS_EMPLOYED', temp)

        return True


if __name__ == "__main__":
    ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
    ROOT_PACKAGE = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer'

    import sys

    sys.path.append(ROOT_PACKAGE)
    from experiment_graph.data_storage import DedupedStorageManager
    from experiment_graph.executor import CollaborativeExecutor
    from experiment_graph.execution_environment import ExecutionEnvironment
    from experiment_graph.optimizations.Reuse import FastBottomUpReuse
    from experiment_graph.materialization_algorithms.materialization_methods import AllMaterializer

    workload = fork_join_start_here_intro()

    mat_budget = 16.0 * 1024.0 * 1024.0
    sa_materializer = AllMaterializer(storage_budget=mat_budget)

    ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=FastBottomUpReuse.NAME)

    root_data = ROOT + '/data'
    # database_path = \
    #     root_data + '/experiment_graphs/kaggle_home_credit/start_here_a_gentle_introduction/all_mat'
    # if os.path.exists(database_path):
    #     ee.load_history_from_disk(database_path)

    executor = CollaborativeExecutor(ee, sa_materializer)
    execution_start = datetime.now()

    executor.end_to_end_run(workload=workload, root_data=root_data, verbose=1)
    # executor.store_experiment_graph(database_path)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    print('finished execution in {} seconds'.format(elapsed))
