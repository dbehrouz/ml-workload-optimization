#!/usr/bin/env python

"""Optimized Fork of Workload 2

This script is the optimized version of the workload 'fork_introduction_to_manual_feature_engineering'
which utilizes our Experiment Graph for optimizing the workload
"""
import warnings
# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib
import pandas as pd

from experiment_graph.workload import Workload

matplotlib.use('ps')
# numpy and pandas for data manipulation

# Experiment Graph

# Suppress warnings

warnings.filterwarnings('ignore')


class fork_introduction_to_manual_feature_engineering(Workload):

    def run(self, execution_environment, root_data, verbose=0):

        def agg_numeric(df, group_var, df_name):
            """Aggregates the numeric values in a dataframe. This can
            be used to create features for each instance of the grouping variable.

            Parameters
            --------
                df (dataframe):
                    the dataframe to calculate the statistics on
                group_var (string):
                    the variable by which to group df
                df_name (string):
                    the variable used to rename the columns

            Return
            --------
                agg (dataframe):
                    a dataframe with the statistics aggregated for
                    all numeric columns. Each instance of the grouping variable will have
                    the statistics (mean, min, max, sum; currently supported) calculated.
                    The columns are also renamed to keep track of features created.

            """
            df_columns = df.data(verbose=verbose).columns
            # Remove id variables other than grouping variable
            for col in df_columns:
                if col != group_var and 'SK_ID' in col:
                    df = df.drop(columns=col)

            numeric_df = df.select_dtypes('number')

            # Group by the specified variable and calculate the statistics
            agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum'])

            # Need to create new column names
            column_names = [group_var]
            columns = agg.data(verbose=verbose).columns
            for c in columns:
                if c != group_var:
                    column_names.append('{}_{}'.format(df_name, c))
            return agg.set_columns(column_names)

        def count_categorical(df, group_var, df_name):
            """Computes counts and normalized counts for each observation
            of `group_var` of each unique category in every categorical variable

            Parameters
            --------
            df : dataframe
                The dataframe to calculate the value counts for.

            group_var : string
                The variable by which to group the dataframe. For each unique
                value of this variable, the final dataframe will have one row

            df_name : string
                Variable added to the front of column names to keep track of columns


            Return
            --------
            categorical : dataframe
                A dataframe with counts and normalized counts of each unique category in every categorical variable
                with one row for every unique value of the `group_var`.

            """

            # Select the categorical columns
            categorical = df.select_dtypes('object').onehot_encode()

            # Make sure to put the identifying id on the column
            categorical = categorical.add_columns(group_var, df[group_var])

            # Groupby the group var and calculate the sum and mean
            categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

            column_names = [group_var]
            # Need to create new column names
            columns = categorical.data(verbose=verbose).columns
            for c in columns:
                if c != group_var:
                    column_names.append('{}_{}'.format(df_name, c))

            return categorical.set_columns(column_names)

        # Read in new copies of all the dataframes
        train = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        bureau = execution_environment.load(root_data + '/kaggle_home_credit/bureau.csv')
        bureau_balance = execution_environment.load(root_data + '/kaggle_home_credit/bureau_balance.csv')

        bureau_counts = count_categorical(bureau, group_var='SK_ID_CURR', df_name='bureau')
        bureau_counts.head().data(verbose=verbose)

        bureau_agg = agg_numeric(bureau.drop(columns=['SK_ID_BUREAU']), group_var='SK_ID_CURR', df_name='bureau')
        bureau_agg.head().data(verbose=verbose)

        bureau_balance_counts = count_categorical(bureau_balance, group_var='SK_ID_BUREAU', df_name='bureau_balance')
        bureau_balance_counts.head().data(verbose=verbose)

        bureau_balance_agg = agg_numeric(bureau_balance, group_var='SK_ID_BUREAU', df_name='bureau_balance')
        bureau_balance_agg.head().data(verbose=1)

        # Dataframe grouped by the loan
        bureau_by_loan = bureau_balance_agg.merge(
            bureau_balance_counts, on='SK_ID_BUREAU', how='outer')

        # Merge to include the SK_ID_CURR
        bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(
            bureau_by_loan, on='SK_ID_BUREAU', how='left')

        # Aggregate the stats for each client
        bureau_balance_by_client = agg_numeric(
            bureau_by_loan.drop(columns=['SK_ID_BUREAU']),
            group_var='SK_ID_CURR',
            df_name='client')

        original_features = list(train.data(verbose=verbose).columns)
        print('Original Number of Features: ', len(original_features))

        # Merge with the value counts of bureau
        train = train.merge(bureau_counts, on='SK_ID_CURR', how='left')

        # Merge with the stats of bureau
        train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')

        # Merge with the monthly information grouped by client
        train = train.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')

        new_features = list(train.data(verbose=verbose).columns)
        print('Number of features using previous loans from other institutions data: ', len(new_features))

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
    from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer

    workload = fork_introduction_to_manual_feature_engineering()

    mat_budget = 16.0 * 1024.0 * 1024.0
    sa_materializer = StorageAwareMaterializer(storage_budget=mat_budget)

    ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=FastBottomUpReuse.NAME)

    root_data = ROOT + '/data'
    # database_path = \
    #     root_data + '/experiment_graphs/kaggle_home_credit/introduction_to_manual_feature_engineering/sa_16'
    # if os.path.exists(database_path):
    #     ee.load_history_from_disk(database_path)
    executor = CollaborativeExecutor(ee, sa_materializer)
    execution_start = datetime.now()

    executor.end_to_end_run(workload=workload, root_data=root_data, verbose=1)
    # executor.store_experiment_graph(database_path)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    print('finished execution in {} seconds'.format(elapsed))
