#!/usr/bin/env python

"""Optimized workload ID = 7

This script is the optimized version of the baseline workload 'fork_introduction_to_manual_feature_engineering_p2'
(ID = 7) which utilizes our Experiment Graph for optimizing the workload

Number of artifacts: 121
Total artifact size: 21 GB
"""
import warnings
# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib

from experiment_graph.workload import Workload

matplotlib.use('ps')
# numpy and pandas for data manipulation

# Experiment Graph

# Suppress warnings
warnings.filterwarnings('ignore')


class fork_introduction_to_manual_feature_engineering_p2(Workload):

    def run(self, execution_environment, root_data, verbose=0):

        def agg_numeric(df, parent_var, df_name):
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
                if col != parent_var and 'SK_ID' in col:
                    df = df.drop(columns=col)

            numeric_df = df.select_dtypes('number')

            # Group by the specified variable and calculate the statistics
            agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

            # Need to create new column names
            column_names = [parent_var]
            columns = agg.data(verbose=verbose).columns
            for c in columns:
                if c != parent_var:
                    column_names.append('{}_{}'.format(df_name, c))
            return agg.set_columns(column_names)

        def agg_categorical(df, parent_var, df_name):
            """
            Aggregates the categorical features in a child dataframe
            for each observation of the parent variable.

            Parameters
            --------
            df : dataframe
                The dataframe to calculate the value counts for.

            parent_var : string
                The variable by which to group and aggregate the dataframe. For each unique
                value of this variable, the final dataframe will have one row

            df_name : string
                Variable added to the front of column names to keep track of columns


            Return
            --------
            categorical : dataframe
                A dataframe with aggregated statistics for each observation of the parent_var
                The columns are also renamed and columns with duplicate values are removed.

            """
            categorical = df.select_dtypes('object').onehot_encode()

            categorical = categorical.add_columns(parent_var, df[parent_var])

            # Groupby the group var and calculate the sum and mean
            categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])

            column_names = [parent_var]
            columns = categorical.data(verbose=verbose).columns
            for c in columns:
                if c != parent_var:
                    column_names.append('{}_{}'.format(df_name, c))

            return categorical.set_columns(column_names)

        previous = execution_environment.load(root_data + '/kaggle_home_credit/previous_application.csv')
        previous.head().data(verbose=verbose)

        # Calculate aggregate statistics for each numeric column
        previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')
        print('Previous aggregation shape: ', previous_agg.shape().data(verbose=verbose))
        previous_agg.head().data(verbose=verbose)

        # Calculate value counts for each categorical column
        previous_counts = agg_categorical(previous, 'SK_ID_CURR', 'previous')
        print('Previous counts shape: ', previous_counts.shape().data(verbose=verbose))
        previous_counts.head().data(verbose=verbose)

        train = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        test = execution_environment.load(root_data + '/kaggle_home_credit/application_test.csv')

        test_labels = execution_environment.load(root_data + '/kaggle_home_credit/application_test_labels.csv')

        # Merge in the previous information
        train = train.merge(previous_counts, on='SK_ID_CURR', how='left')
        train = train.merge(previous_agg, on='SK_ID_CURR', how='left')

        test = test.merge(previous_counts, on='SK_ID_CURR', how='left')
        test = test.merge(previous_agg, on='SK_ID_CURR', how='left')

        def remove_missing_columns(train, test, threshold=90):

            # Total missing values
            train_miss = train.isnull().sum().data(verbose=verbose)
            train_miss_percent = 100 * train_miss / train.shape().data(verbose=verbose)[0]

            # Total missing values
            test_miss = test.isnull().sum().data(verbose=verbose)
            test_miss_percent = 100 * test_miss / test.shape().data(verbose=verbose)[0]

            # list of missing columns for train and test
            missing_train_columns = list(train_miss.index[train_miss_percent > threshold])
            missing_test_columns = list(test_miss.index[test_miss_percent > threshold])

            # Combine the two lists together
            missing_columns = list(set(missing_train_columns + missing_test_columns))

            # Print information
            print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))

            # Drop the missing columns and return
            train = train.drop(columns=missing_columns)
            test = test.drop(columns=missing_columns)

            return train, test

        train, test = remove_missing_columns(train, test)

        def aggregate_client(df, group_vars, df_names):
            """Aggregate a dataframe with data at the loan level
            at the client level

            Args:
                df (dataframe): data at the loan level
                group_vars (list of two strings): grouping variables for the loan
                and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
                df_names (list of two strings): names to call the resulting columns
                (example ['cash', 'client'])

            Returns:
                df_client (dataframe): aggregated numeric stats at the client level.
                Each client will have a single row with all the numeric data aggregated
            """

            # Aggregate the numeric columns
            df_agg = agg_numeric(df, parent_var=group_vars[0], df_name=df_names[0])

            # If there are categorical variables
            if any(df.dtypes().data(verbose=verbose) == 'category'):

                # Count the categorical columns
                df_counts = agg_categorical(df, parent_var=group_vars[0], df_name=df_names[0])

                # Merge the numeric and categorical
                df_by_loan = df_counts.merge(df_agg, on=group_vars[0], how='outer')

                # Merge to get the client id in dataframe
                df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

                # Remove the loan id
                df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

                # Aggregate numeric stats by column
                df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])

            # No categorical variables
            else:
                # Merge to get the client id in dataframe
                df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

                # Remove the loan id
                df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

                # Aggregate numeric stats by column
                df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])

            return df_by_client

        cash = execution_environment.load(root_data + '/kaggle_home_credit/POS_CASH_balance.csv')
        cash.head()

        cash_by_client = aggregate_client(cash, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['cash', 'client'])
        cash_by_client.head()

        print 'Cash by Client Shape: {}'.format(cash_by_client.shape().data(verbose=verbose))
        train = train.merge(cash_by_client, on='SK_ID_CURR', how='left')
        test = test.merge(cash_by_client, on='SK_ID_CURR', how='left')

        train, test = remove_missing_columns(train, test)

        credit = execution_environment.load(root_data + '/kaggle_home_credit/credit_card_balance.csv')
        credit.head()

        credit_by_client = aggregate_client(credit, group_vars=['SK_ID_PREV', 'SK_ID_CURR'],
                                            df_names=['credit', 'client'])
        credit_by_client.head()

        print 'Credit by client shape: {}'.format(credit_by_client.shape().data(verbose=verbose))

        train = train.merge(credit_by_client, on='SK_ID_CURR', how='left')
        test = test.merge(credit_by_client, on='SK_ID_CURR', how='left')

        train, test = remove_missing_columns(train, test)

        installments = execution_environment.load(root_data + '/kaggle_home_credit/installments_payments.csv')
        installments.head()

        installments_by_client = aggregate_client(installments, group_vars=['SK_ID_PREV', 'SK_ID_CURR'],
                                                  df_names=['installments', 'client'])
        installments_by_client.head()

        print 'Installments by client shape: {}'.format(installments_by_client.shape().data(verbose=verbose))

        train = train.merge(installments_by_client, on='SK_ID_CURR', how='left')
        test = test.merge(installments_by_client, on='SK_ID_CURR', how='left')

        train, test = remove_missing_columns(train, test)

        print 'Final Training Shape: {}'.format(train.shape().data(verbose=verbose))
        print 'Final Testing Shape: {}'.format(test.shape().data(verbose=verbose))

        from experiment_graph.sklearn_helper.sklearn_wrappers import LGBMClassifier

        def model(lgb_featres, test_features, encoding='ohe'):

            """Train and test a light gradient boosting model using
            cross validation.

            Parameters
            --------
                features (Dataset):
                    dataframe of training features to use
                    for training a model. Must include the TARGET column.
                test_features (Dataset):
                    dataframe of testing features to use
                    for making predictions with the model.
                encoding (str, default = 'ohe'):
                    method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label
                    encoding
                    n_folds (int, default = 5): number of folds to use for cross validation

            Return
            --------
                submission (Dataset):
                    dataframe with `SK_ID_CURR` and `TARGET` probabilities
                    predicted by the model.
                feature_importances (Dataset):
                    dataframe with the feature importances from the model.
                valid_metrics (Dataset):
                    dataframe with training and validation metrics (ROC AUC) for each fold and overall.

            """

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
            model = LGBMClassifier(objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)

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
                                custom_args={'num_iteration': best_iteration}).data(verbose=verbose)
            print 'LGBMClassifier with AUC score: {}'.format(score)

        model(train, test)
        # LGBMClassifier with AUC score: {'auc': 0.77630698005361221}

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

    workload = fork_introduction_to_manual_feature_engineering_p2()

    mat_budget = 16.0 * 1024.0 * 1024.0
    sa_materializer = StorageAwareMaterializer(storage_budget=mat_budget)

    ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=FastBottomUpReuse.NAME)

    root_data = ROOT + '/data'
    # database_path = \
    #     root_data + '/experiment_graphs/kaggle_home_credit/introduction_to_manual_feature_engineering_p2/sa_16'
    # if os.path.exists(database_path):
    #     ee.load_history_from_disk(database_path)
    executor = CollaborativeExecutor(ee, materializer=sa_materializer)
    execution_start = datetime.now()

    executor.end_to_end_run(workload=workload, root_data=root_data, verbose=1)
    # executor.store_experiment_graph(database_path)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    print('finished execution in {} seconds'.format(elapsed))
