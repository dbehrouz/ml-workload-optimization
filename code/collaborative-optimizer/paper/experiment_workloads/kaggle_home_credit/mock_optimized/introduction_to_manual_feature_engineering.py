#!/usr/bin/env python

"""Optimized workload 2

This script is the optimized version of the workload 'introduction_to_manual_feature_engineering'
which utilizes our Experiment Graph for optimizing the workload
"""
import os
import warnings
# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib
import pandas as pd

from experiment_graph.workload import Workload

matplotlib.use('ps')
import matplotlib.pyplot as plt
# numpy and pandas for data manipulation
import seaborn as sns

# Experiment Graph

# Suppress warnings

warnings.filterwarnings('ignore')


class introduction_to_manual_feature_engineering(Workload):

    def run(self, execution_environment, root_data, verbose=0):
        # Read in bureau
        bureau = execution_environment.load(root_data + '/kaggle_home_credit/bureau.csv')

        bureau.head().data(verbose=verbose)
        previous_loan_counts = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count().rename(
            columns={'SK_ID_BUREAU': 'previous_loan_counts'})
        # previous_loan_counts = previous_loan_counts.rename(columns=['SK_ID_CURR', 'previous_loan_counts'])
        previous_loan_counts.head().data(verbose=verbose)

        # Join to the training dataframe
        train = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        train = train.merge(previous_loan_counts, on='SK_ID_CURR', how='left')

        train = train.replace_columns('previous_loan_counts', train['previous_loan_counts'].fillna(0))
        train.head().data(verbose=verbose)

        # Plots the distribution of a variable colored by value of the target
        def kde_target(var_name, df):
            # Calculate the correlation coefficient between the new variable and the target
            corr = df['TARGET'].corr(df[var_name])

            # Calculate medians for repaid vs not repaid
            avg_repaid = df[df['TARGET'] == 0][var_name].median()
            avg_not_repaid = df[df['TARGET'] == 1][var_name].median()

            plt.figure(figsize=(12, 6))

            # Plot the distribution for target == 0 and target == 1
            sns.kdeplot(df[df['TARGET'] == 0][var_name].dropna().data(verbose=verbose), label='TARGET == 0')
            sns.kdeplot(df[df['TARGET'] == 1][var_name].dropna().data(verbose=verbose), label='TARGET == 1')

            # label the plot
            plt.xlabel(var_name)
            plt.ylabel('Density')
            plt.title('%s Distribution' % var_name)
            plt.legend()
            # print out the correlation
            print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr.data(verbose=verbose)))
            # Print out average values
            print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid.data(verbose=verbose))
            print('Median value for loan that was repaid =     %0.4f' % avg_repaid.data(verbose=verbose))

        kde_target('EXT_SOURCE_3', train)

        kde_target('previous_loan_counts', train)

        # Group by the client id, calculate aggregation statistics
        bureau_agg = bureau.drop(columns=['SK_ID_BUREAU']).groupby('SK_ID_CURR').agg(
            ['count', 'mean', 'max', 'min', 'sum'])
        columns = []
        bureau_agg_cols = bureau_agg.data(verbose=verbose).columns
        for c in bureau_agg_cols:
            if c != 'SK_ID_CURR':
                columns.append('bureau_{}'.format(c))
            else:
                columns.append(c)
        bureau_agg = bureau_agg.set_columns(columns)
        bureau_agg.head().data(verbose=verbose)

        # Merge with the training data
        train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')
        train.head().data(verbose=verbose)

        # List of new correlations
        new_corrs = []
        columns = bureau_agg.data(verbose=verbose).columns
        # Iterate through the columns
        for col in columns:
            # Calculate correlation with the target
            corr = train['TARGET'].corr(train[col])

            # Append the list as a tuple

            new_corrs.append((col, corr.data(verbose=verbose)))

        # Sort the correlations by the absolute value
        # Make sure to reverse to put the largest values at the front of list
        new_corrs = sorted(new_corrs, key=lambda x: abs(x[1]), reverse=True)
        new_corrs[:15]

        kde_target('bureau_DAYS_CREDIT_mean', train)
        return True


if __name__ == "__main__":
    ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
    ROOT_PACKAGE = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer'

    import sys

    sys.path.append(ROOT_PACKAGE)
    from experiment_graph.data_storage import DedupedStorageManager
    from experiment_graph.executor import CollaborativeExecutor
    from experiment_graph.execution_environment import ExecutionEnvironment
    from experiment_graph.optimizations.Reuse import LinearTimeReuse
    from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer

    workload = introduction_to_manual_feature_engineering()

    mat_budget = 16.0 * 1024.0 * 1024.0
    sa_materializer = StorageAwareMaterializer(storage_budget=mat_budget)

    ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=LinearTimeReuse.NAME)

    root_data = ROOT + '/data'
    # database_path = \
    #     root_data + '/experiment_graphs/kaggle_home_credit/introduction_to_manual_feature_engineering/sa_16'
    # if os.path.exists(database_path):
    #     ee.load_history_from_disk(database_path)
    executor = CollaborativeExecutor(execution_environment=ee, materializer=sa_materializer)
    execution_start = datetime.now()

    executor.end_to_end_run(workload=workload, root_data=root_data, verbose=1)

    # executor.store_experiment_graph(database_path)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    print('finished execution in {} seconds'.format(elapsed))
