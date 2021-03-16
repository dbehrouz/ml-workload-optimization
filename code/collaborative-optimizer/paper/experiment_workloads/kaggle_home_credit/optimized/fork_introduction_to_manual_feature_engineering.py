#!/usr/bin/env python

"""Optimized workload ID = 6

This script is the optimized version of the baseline workload 'fork_introduction_to_manual_feature_engineering'
(ID = 6) which utilizes our Experiment Graph for optimizing the workload

Number of artifacts: 121
Total artifact size: 21 GB
"""
import warnings
# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib
import pandas as pd

from experiment_graph.data_storage import DedupedStorageManager
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.executor import CollaborativeExecutor
from experiment_graph.materialization_algorithms.materialization_methods import StorageAwareMaterializer
from experiment_graph.optimizations.Reuse import LinearTimeReuse
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

        # Function to calculate missing values by column# Funct
        def missing_values_table(dataset):
            # Total missing values
            mis_val = dataset.isnull().sum().data(verbose=verbose)

            mis_val_percent = 100 * mis_val / len(dataset.data(verbose=verbose))

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
            print("Your selected dataframe has " + str(dataset.shape().data(verbose=verbose)[1]) + " columns.\n"
                                                                                                   "There are " + str(
                mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")

            # Return the dataframe with missing information
            return mis_val_table_ren_columns

        missing_train = missing_values_table(train)
        missing_train.head(10)

        missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > 90])
        len(missing_train_vars)

        # Read in the test dataframe
        test = execution_environment.load(root_data + '/kaggle_home_credit/application_test.csv')

        test_labels = execution_environment.load(root_data + '/kaggle_home_credit/application_test_labels.csv')

        # Merge with the value counts of bureau
        test = test.merge(bureau_counts, on='SK_ID_CURR', how='left')

        # Merge with the stats of bureau
        test = test.merge(bureau_agg, on='SK_ID_CURR', how='left')

        # Merge with the value counts of bureau balance
        test = test.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')

        print('Shape of Testing Data: ', test.shape().data(verbose=verbose))

        train_labels = train['TARGET']

        # Align the dataframes, this will remove the 'TARGET' column
        train = train.align(test)
        test = test.align(train)

        train = train.add_columns('TARGET', train_labels)

        print('Training Data Shape: ', train.shape().data(verbose=verbose))
        print('Testing Data Shape: ', test.shape().data(verbose=verbose))

        missing_test = missing_values_table(test)
        missing_test.head(10)

        missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > 90])
        len(missing_test_vars)

        missing_columns = list(set(missing_test_vars + missing_train_vars))
        print('There are %d columns with more than 90%% missing in either the training or testing data.' % len(
            missing_columns))

        # Drop the missing columns
        train = train.drop(columns=missing_columns)
        test = test.drop(columns=missing_columns)

        # Calculate all correlations in dataframe
        corrs = train.corr().data(verbose=verbose)

        corrs = corrs.sort_values('TARGET', ascending=False)

        # Ten most positive correlations
        pd.DataFrame(corrs['TARGET'].head(10))

        # Ten most negative correlations
        pd.DataFrame(corrs['TARGET'].dropna().tail(10))

        # kde_target(var_name='bureau_balance_counts_mean', df=train)

        # kde_target(var_name='bureau_CREDIT_ACTIVE_Active_count_norm', df=train)

        # Set the threshold
        threshold = 0.8

        # Empty dictionary to hold correlated variables
        above_threshold_vars = {}

        # For each column, record the variables that are above the threshold
        for col in corrs:
            above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])

        # Track columns to remove and columns already examined
        cols_to_remove = []
        cols_seen = []
        cols_to_remove_pair = []

        # Iterate through columns and correlated columns
        for key, value in above_threshold_vars.items():
            # Keep track of columns already examined
            cols_seen.append(key)
            for x in value:
                if x == key:
                    next
                else:
                    # Only want to remove one in a pair
                    if x not in cols_seen:
                        cols_to_remove.append(x)
                        cols_to_remove_pair.append(key)

        cols_to_remove = list(set(cols_to_remove))
        print('Number of columns to remove: ', len(cols_to_remove))

        train_corrs_removed = train.drop(columns=cols_to_remove)
        test_corrs_removed = test.drop(columns=cols_to_remove)

        print('Training Corrs Removed Shape: ', train_corrs_removed.shape().data(verbose=verbose))
        print('Testing Corrs Removed Shape: ', test_corrs_removed.shape().data(verbose=verbose))

        from experiment_graph.sklearn_helper.sklearn_wrappers import LGBMClassifier

        def model(lgb_featres, test_features, encoding='ohe'):

            """Train and test a light gradient boosting model using
            cross validation.

            Parameters
            --------
                features (pd.DataFrame):
                    dataframe of training features to use
                    for training a model. Must include the TARGET column.
                test_features (pd.DataFrame):
                    dataframe of testing features to use
                    for making predictions with the model.
                encoding (str, default = 'ohe'):
                    method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label
                    encoding
                    n_folds (int, default = 5): number of folds to use for cross validation

            Return
            --------
                submission (pd.DataFrame):
                    dataframe with `SK_ID_CURR` and `TARGET` probabilities
                    predicted by the model.
                feature_importances (pd.DataFrame):
                    dataframe with the feature importances from the model.
                valid_metrics (pd.DataFrame):
                    dataframe with training and validation metrics (ROC AUC) for each fold and overall.

            """

            # Extract the ids
            train_ids = lgb_featres['SK_ID_CURR']
            test_ids = test_features['SK_ID_CURR']

            # Extract the labels for training
            labels = lgb_featres['TARGET']

            # Remove the ids and target
            lgb_featres = lgb_featres.drop(columns=['SK_ID_CURR', 'TARGET'])
            test_features = test_features.drop(columns=['SK_ID_CURR'])

            # One Hot Encoding
            if encoding == 'ohe':
                lgb_featres = lgb_featres.onehot_encode()
                test_features = test_features.onehot_encode()

                features_columns = lgb_featres.data(verbose=verbose).columns
                test_features_columns = test_features.data(verbose=verbose).columns
                for c in features_columns:
                    if c not in test_features_columns:
                        lgb_featres = lgb_featres.drop(c)

                # No categorical indices to record
                cat_indices = 'auto'

            # Catch error if label encoding scheme is not valid
            else:
                raise ValueError("Encoding must be either 'ohe' or 'le'")

            print('Training Data Shape: ', lgb_featres.shape().data(verbose=verbose))
            print('Testing Data Shape: ', test_features.shape().data(verbose=verbose))

            # Extract feature names
            feature_names = list(lgb_featres.data(verbose=verbose).columns)

            # Create the model
            model = LGBMClassifier(
                learning_rate=0.1,
                subsample=.8,
                max_depth=7,
                reg_alpha=.1,
                reg_lambda=.1,
                min_split_gain=.01,
                min_child_weight=2
            )

            # Train the model
            model.fit(lgb_featres, labels, custom_args={'eval_metric': 'auc',
                                                        'categorical_feature': cat_indices,
                                                        'verbose': 200})

            # Record the best iteration
            best_iteration = model.best_iteration()

            # Make predictions
            score = model.score(test_features,
                                test_labels['TARGET'],
                                score_type='auc',
                                custom_args={'num_iteration': best_iteration}).data()

            print('LGBMClassifier with AUC score: {}'.format(score))

        train_control = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        test_control = execution_environment.load(root_data + '/kaggle_home_credit/application_test.csv')

        fi = model(train_control, test_control)

        fi_raw = model(train, test)

        fi_corrs = model(train_corrs_removed, test_corrs_removed)

        return True


if __name__ == "__main__":
    ROOT = '/Users/bede01/Documents/work/phd-papers/published/ml-workload-optimization'
    root_data = ROOT + '/data'

    workload = fork_introduction_to_manual_feature_engineering()

    mat_budget = 16.0 * 1024.0 * 1024.0
    sa_materializer = StorageAwareMaterializer(storage_budget=mat_budget)

    ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=LinearTimeReuse.NAME)

    # database_path = \
    #     root_data + '/experiment_graphs/kaggle_home_credit/introduction_to_manual_feature_engineering/sa_16'
    # if os.path.exists(database_path):
    #     ee.load_history_from_disk(database_path)
    executor = CollaborativeExecutor(ee, materializer=sa_materializer)
    execution_start = datetime.now()

    executor.end_to_end_run(workload=workload, root_data=root_data, verbose=1)
    # executor.store_experiment_graph(database_path)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    print('finished execution in {} seconds'.format(elapsed))
