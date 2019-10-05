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

        app_train["DAYS_EMPLOYED"].data(verbose).plot.hist(title='Days Employment Histogram');
        plt.xlabel('Days Employment')

        days_employed_anom = app_test["DAYS_EMPLOYED"] == 365243
        app_test = app_test.add_columns('DAYS_EMPLOYED_ANOM', days_employed_anom)
        temp = app_test['DAYS_EMPLOYED'].replace({365243: np.nan})
        app_test = app_test.drop('DAYS_EMPLOYED')
        app_test = app_test.add_columns('DAYS_EMPLOYED', temp)
        print('There are %d anomalies in the test data out of %d entries'
              % (app_test['DAYS_EMPLOYED_ANOM'].sum().data(verbose),
                 app_test.shape().data(verbose)[0]))

        correlations = app_train.corr().data(verbose)
        top = correlations['TARGET'].sort_values()
        # Display correlations
        print('Most Positive Correlations:\n', top.tail(15))
        print('\nMost Negative Correlations:\n', top.head(15))

        abs_age = app_train['DAYS_BIRTH'].abs()
        app_train = app_train.drop('DAYS_BIRTH')
        app_train = app_train.add_columns('DAYS_BIRTH', abs_age)
        app_train['DAYS_BIRTH'].corr(app_train['TARGET']).data(verbose)

        # Set the style of plots
        plt.style.use('fivethirtyeight')

        # Plot the distribution of ages in years
        plt.hist((app_train['DAYS_BIRTH'] / 365).data(verbose), edgecolor='k', bins=25)
        plt.title('Age of Client')
        plt.xlabel('Age (years)')
        plt.ylabel('Count')

        plt.figure(figsize=(10, 8))
        # KDE plot of loans that were repaid on time
        sns.kdeplot((app_train[app_train['TARGET'] == 0]['DAYS_BIRTH'] / 365).data(verbose), label='target == 0')
        # KDE plot of loans which were not repaid on time
        sns.kdeplot((app_train[app_train['TARGET'] == 1]['DAYS_BIRTH'] / 365).data(verbose), label='target == 1')
        # Labeling of plot
        plt.xlabel('Age (years)')
        plt.ylabel('Density')
        plt.title('Distribution of Ages')

        # Age information into a separate dataframe
        age_data = app_train[['TARGET', 'DAYS_BIRTH']]
        years_birth = age_data['DAYS_BIRTH'] / 365
        age_data = age_data.add_columns('YEARS_BIRTH', years_birth)
        binned = age_data['YEARS_BIRTH'].binning(20, 70, 11)
        binned.setname('YEARS_BINNED')
        age_data = age_data.add_columns('YEARS_BINNED', binned)
        age_data.head(10).data(verbose)

        age_groups = age_data.groupby('YEARS_BINNED').mean()
        age_groups.data(verbose)

        plt.figure(figsize=(8, 8))

        # Graph the age bins and the average of the target as a bar plot
        plt.bar(age_groups.data(verbose).index.astype(str), age_groups.data(verbose)['TARGET'] * 100)

        # Plot labeling
        plt.xticks(rotation=75)
        plt.xlabel('Age Group (years)')
        plt.ylabel('Failure to Repay (%)')
        plt.title('Failure to Repay by Age Group')

        ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
        ext_data_corrs = ext_data.corr().data(verbose)

        plt.figure(figsize=(8, 6))

        # Heatmap of correlations
        sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
        plt.title('Correlation Heatmap')

        plt.figure(figsize=(10, 12))

        # iterate through the sources
        for i, column in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
            # create a new subplot for each source
            plt.subplot(3, 1, i + 1)
            # plot repaid loans
            source_data = app_train[[column, 'TARGET']][app_train['TARGET'] == 0]
            sns.kdeplot(source_data[app_train[column].notna()][column].data(verbose), label='target == 0')
            # plot loans that were not repaid
            source_data = app_train[[column, 'TARGET']][app_train['TARGET'] == 1]
            sns.kdeplot(source_data[app_train[column].notna()][column].data(verbose), label='target == 1')

            # Label the plots
            plt.title('Distribution of %s by Target Value' % column)
            plt.xlabel('%s' % column)
            plt.ylabel('Density')

        plt.tight_layout(h_pad=2.5)

        # Copy the data for plotting
        plot_data = ext_data.drop('DAYS_BIRTH')

        # Add in the age of the client in years
        plot_data = plot_data.add_columns('YEARS_BIRTH', age_data['YEARS_BIRTH'])
        # Drop na values and limit to first 100000 rows
        plot_data = plot_data.head(100000).dropna()

        # Create the pair grid object
        grid = sns.PairGrid(data=plot_data.data(verbose), size=3, diag_sharey=False,
                            hue='TARGET',
                            vars=[x for x in list(plot_data.data(verbose).columns) if x != 'TARGET'])

        # Upper is a scatter plot
        grid.map_upper(plt.scatter, alpha=0.2)

        # Diagonal is a histogram
        grid.map_diag(sns.kdeplot)

        # Bottom is density plot
        grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r)

        plt.suptitle('Ext Source and Age Features Pairs Plot', size=32, y=1.05)

        app_train_domain = app_train.copy()
        app_test_domain = app_test.copy()

        app_train_domain = app_train_domain.add_columns('CREDIT_INCOME_PERCENT',
                                                        app_train_domain['AMT_CREDIT'] / app_train_domain[
                                                            'AMT_INCOME_TOTAL'])
        app_train_domain = app_train_domain.add_columns('ANNUITY_INCOME_PERCENT',
                                                        app_train_domain['AMT_ANNUITY'] / app_train_domain[
                                                            'AMT_INCOME_TOTAL'])
        app_train_domain = app_train_domain.add_columns('CREDIT_TERM',
                                                        app_train_domain['AMT_ANNUITY'] / app_train_domain[
                                                            'AMT_CREDIT'])
        app_train_domain = app_train_domain.add_columns('DAYS_EMPLOYED_PERCENT',
                                                        app_train_domain['DAYS_EMPLOYED'] / app_train_domain[
                                                            'DAYS_BIRTH'])

        app_test_domain = app_test_domain.add_columns('CREDIT_INCOME_PERCENT',
                                                      app_test_domain['AMT_CREDIT'] / app_test_domain[
                                                          'AMT_INCOME_TOTAL'])
        app_test_domain = app_test_domain.add_columns('ANNUITY_INCOME_PERCENT',
                                                      app_test_domain['AMT_ANNUITY'] / app_test_domain[
                                                          'AMT_INCOME_TOTAL'])
        app_test_domain = app_test_domain.add_columns('CREDIT_TERM',
                                                      app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT'])
        app_test_domain = app_test_domain.add_columns('DAYS_EMPLOYED_PERCENT',
                                                      app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH'])

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
                    method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
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

                features_columns = lgb_featres.data(verbose).columns
                test_features_columns = test_features.data(verbose).columns
                for c in features_columns:
                    if c not in test_features_columns:
                        lgb_featres = lgb_featres.drop(c)

                # No categorical indices to record
                cat_indices = 'auto'
            else:
                raise ValueError("Encoding must be either 'ohe' or 'le'")

            print('Training Data Shape: ', lgb_featres.shape().data(verbose))
            print('Testing Data Shape: ', test_features.shape().data(verbose))

            # Extract feature names
            feature_names = list(lgb_featres.data(verbose).columns)

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
                                custom_args={'num_iteration': best_iteration}).data(verbose)
            print 'LGBMClassifier with AUC score: {}'.format(score)


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
        model(app_train_domain, app_test_domain)
        # LGBMClassifier with AUC score: {'auc': 0.62983905432040155}

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
