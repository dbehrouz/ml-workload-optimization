#!/usr/bin/env python

"""Optimized Fork of Workload 1

This script is the optimized version of the fork of workload 1 submitted by user taozhongxiao
which utilizes our Experiment Graph for optimizing the workload.
"""
import os
import warnings
import pandas as pd
# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib

matplotlib.use('ps')
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from experiment_graph.workload import Workload

matplotlib.use('ps')

import numpy as np
# numpy and pandas for data manipulation
from experiment_graph.sklearn_helper.sklearn_wrappers import LGBMClassifier

# Experiment Graph

# Suppress warnings
warnings.filterwarnings('ignore')


class fork_taozhongxiao_start_here_a_gentle_introduction(Workload):

    def run(self, execution_environment, root_data, verbose=0):
        MAX_EVALS = 5
        print(os.listdir(root_data))
        train_features = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        train_features = train_features.sample(n=16000, random_state=42)
        print(train_features.shape().data(verbose=verbose))

        train_features.dtypes().data(verbose=verbose).value_counts()

        train_features = train_features.select_dtypes('number')
        train_labels = train_features['TARGET']
        train_features = train_features.drop(columns=['TARGET', 'SK_ID_CURR'])

        test_features = execution_environment.load(root_data + '/kaggle_home_credit/application_test.csv')
        test_features.head().data(verbose=verbose)
        test_labels = execution_environment.load(root_data + '/kaggle_home_credit/application_test_labels.csv')
        test_features = test_features.select_dtypes('number')  # manually added
        test_features = test_features.drop(columns=['SK_ID_CURR'])  # manually added

        print("Training features shape: ", train_features.shape().data(verbose=verbose))
        print("Testing features shape: ", test_features.shape().data(verbose=verbose))

        model = LGBMClassifier()
        default_params = model.get_params()
        model.fit(train_features, train_labels)

        score = model.score(test_features,
                            test_labels['TARGET'],
                            score_type='auc').data(verbose=verbose)

        print 'LGBMClassifier with AUC score: {}'.format(score)

        def objective(hyperparameters, iteration):
            if 'n_estimators' in hyperparameters.keys():
                del hyperparameters['n_estimators']
            model = LGBMClassifier(**hyperparameters)
            model.fit(train_features, train_labels)
            score = model.score(test_features,
                                test_labels['TARGET'],
                                score_type='auc').data(verbose=verbose)
            return [score, hyperparameters, iteration]

        score, params, iteration = objective(default_params, 1)

        print('The cross-validation ROC AUC was {}.'.format(score))

        param_grid = {
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'num_leaves': list(range(20, 150)),
            'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
            'subsample_for_bin': list(range(20000, 300000, 20000)),
            'min_child_samples': list(range(20, 500, 5)),
            'reg_alpha': list(np.linspace(0, 1)),
            'reg_lambda': list(np.linspace(0, 1)),
            'colsample_bytree': list(np.linspace(0.6, 1, 10)),
            'subsample': list(np.linspace(0.5, 1, 100)),
            'is_unbalance': [True, False]
        }

        import random

        random.seed(50)

        # Randomly sample a boosting type
        boosting_type = random.sample(param_grid['boosting_type'], 1)[0]

        # Set subsample depending on boosting type
        subsample = 1.0 if boosting_type == 'goss' else random.sample(param_grid['subsample'], 1)[0]

        print('Boosting type: ', boosting_type)
        print('Subsample ratio: ', subsample)

        def grid_search(param_grid, max_evals=MAX_EVALS):
            """Grid search algorithm (with limit on max evals)"""
            results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                                   index=list(range(MAX_EVALS)))
            keys, values = zip(*param_grid.items())
            i = 0
            for v in itertools.product(*values):
                hyperparameters = dict(zip(keys, v))
                hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                    'subsample']
                eval_results = objective(hyperparameters, i)
                results.loc[i, :] = eval_results
                i += 1
                if i > max_evals:
                    break
            results.sort_values('score', ascending=False, inplace=True)
            results.reset_index(inplace=True)
            return results

        grid_results = grid_search(param_grid)
        print 'The best validation score was {}'.format(grid_results.loc[0, 'score'])
        print '\nThe best hyperparameters were:'
        import pprint
        pprint.pprint(grid_results.loc[0, 'params'])

        grid_search_params = grid_results.loc[0, 'params']
        grid_search_params['random_state'] = 42
        model = LGBMClassifier(**grid_search_params)
        model.fit(train_features, train_labels)

        score = model.score(test_features,
                            test_labels['TARGET'],
                            score_type='auc').data(verbose=verbose)

        print 'The best model from grid search scores {} ROC AUC on the test set.'.format(score)

        random.seed(50)

        # Randomly sample from dictionary
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        # Deal with subsample ratio
        random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

        random_params

        # Function to calculate missing values by column# Funct
        def random_search(param_grid, max_evals=MAX_EVALS):
            """Random search for hyperparameter optimization"""

            # Dataframe for results
            results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                                   index=list(range(MAX_EVALS)))

            # Keep searching until reach max evaluations
            for i in range(max_evals):
                # Choose random hyperparameters
                hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
                hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                    'subsample']

                # Evaluate randomly selected hyperparameters
                eval_results = objective(hyperparameters, i)

                results.loc[i, :] = eval_results

            # Sort with best score on top
            results.sort_values('score', ascending=False, inplace=True)
            results.reset_index(inplace=True)
            return results

        random_results = random_search(param_grid)

        print('The best validation score was {}'.format(random_results.loc[0, 'score']))
        print('\nThe best hyperparameters were:')

        import pprint
        pprint.pprint(random_results.loc[0, 'params'])

        random_search_params = random_results.loc[0, 'params']

        # Create, train, test model
        grid_search_params['random_state'] = 42
        model = LGBMClassifier(**random_search_params)
        model.fit(train_features, train_labels)

        score = model.score(test_features,
                            test_labels['TARGET'],
                            score_type='auc').data(verbose=verbose)

        print 'The best model from random search scores {} ROC AUC on the test set.'.format(score)

        train = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        test = execution_environment.load(root_data + '/kaggle_home_credit/application_test.csv')

        test_ids = test['SK_ID_CURR']
        train_labels = train['TARGET']

        train = train.drop(columns=['SK_ID_CURR', 'TARGET'])
        test = test.drop(columns=['SK_ID_CURR'])

        print('Training shape: ', train.shape().data(verbose=verbose))
        print('Testing shape: ', test.shape().data(verbose=verbose))

        from experiment_graph.sklearn_helper.preprocessing import LabelEncoder
        # Create a label encoder object
        le_count = 0

        columns = train.select_dtypes('object').data(verbose).columns
        for col in columns:
            # we are not using nunique because it discard nan
            if train[col].nunique(dropna=False).data(verbose) <= 2:
                le = LabelEncoder()
                le.fit(train[col])

                train = train.replace_columns(col, le.transform(train[col]))
                test = test.replace_columns(col, le.transform(test[col]))

                # Keep track of how many columns were label encoded
                le_count += 1
        print('%d columns were label encoded.' % le_count)

        train = train.onehot_encode()
        test = test.onehot_encode()

        train_columns = train.data(verbose).columns
        test_columns = test.data(verbose).columns
        for c in train_columns:
            if c not in test_columns:
                train = train.drop(c)

        print('Training Features shape: ', train.shape().data(verbose))
        print('Testing Features shape: ', test.shape().data(verbose))

        hyperparameters = dict(**random_results.loc[0, 'params'])

        model = LGBMClassifier(**hyperparameters)
        model.fit(train, train_labels)

        # Predictions on the test data
        score = model.score(test,
                            test_labels['TARGET'],
                            score_type='auc').data(verbose=verbose)
        print('LGBM score with best hyperparameters from random search {}.'.format(score))

        app_train = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        print('Training data shape: ', app_train.shape().data(verbose=verbose))
        app_train.head().data(verbose=verbose)

        app_test = execution_environment.load(root_data + '/kaggle_home_credit/application_test.csv')
        print('Training data shape: ', app_test.shape().data(verbose=verbose))
        app_test.head().data(verbose=verbose)

        print app_train['TARGET'].value_counts()

        plt.figure(figsize=(10, 5))
        sns.set(style="whitegrid", font_scale=1)
        g = sns.distplot(app_train['TARGET'].data(verbose=verbose), kde=False,
                         hist_kws={"alpha": 1, "color": "#DA1A32"})
        plt.title('Distribution of target (1:default, 0:no default)', size=15)
        plt.show()

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

        # Make a new dataframe for polynomial features
        poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
        poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

        # imputer for handling missing values
        from experiment_graph.sklearn_helper.preprocessing import Imputer

        imputer = Imputer(strategy='median')

        poly_target = poly_features['TARGET']

        poly_features = poly_features.drop(columns=['TARGET'])

        # Need to impute missing values
        poly_features = imputer.fit_transform(poly_features)
        poly_features_test = imputer.transform(poly_features_test)

        from experiment_graph.sklearn_helper.preprocessing import PolynomialFeatures

        # Create the polynomial object with specified degree
        poly_transformer = PolynomialFeatures(degree=3)

        # Train the polynomial features
        poly_transformer.fit(poly_features)

        # Transform the features
        poly_features = poly_transformer.transform(poly_features)
        poly_features_test = poly_transformer.transform(poly_features_test)
        print('Polynomial Features shape: ', poly_features.shape().data(verbose))

        new_names = poly_transformer.get_feature_names(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                       'EXT_SOURCE_3', 'DAYS_BIRTH'])

        poly_features = poly_features.set_columns(new_names)

        # Add in the target
        poly_features = poly_features.add_columns('TARGET', poly_target)

        # Find the correlations with the target
        poly_corrs = poly_features.corr().data(verbose)['TARGET'].sort_values()

        # Display most negative and most positive
        print(poly_corrs.head(10))
        print(poly_corrs.tail(5))

        poly_features_test = poly_features_test.set_columns(new_names)

        # Merge polynomial features into training dataframe
        poly_features = poly_features.add_columns('SK_ID_CURR', app_train['SK_ID_CURR'])
        app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')

        # Merge polnomial features into testing dataframe
        poly_features_test = poly_features_test.add_columns('SK_ID_CURR', app_test['SK_ID_CURR'])
        app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

        # Align the dataframes
        train_columns = app_train_poly.data(verbose).columns
        test_columns = app_test_poly.data(verbose).columns
        for c in train_columns:
            if c not in test_columns:
                app_train_poly = app_train_poly.drop(c)

        # Print out the new shapes
        print('Training data with polynomial features shape: ',
              app_train_poly.shape().data(verbose))
        print('Testing data with polynomial features shape:  ',
              app_test_poly.shape().data(verbose))

        app_train_poly = app_train_poly.add_columns('CREDIT_INCOME_PERCENT',
                                                    app_train['AMT_CREDIT'] / app_train[
                                                        'AMT_INCOME_TOTAL'])
        app_train_poly = app_train_poly.add_columns('ANNUITY_INCOME_PERCENT',
                                                    app_train['AMT_ANNUITY'] / app_train[
                                                        'AMT_INCOME_TOTAL'])
        app_train_poly = app_train_poly.add_columns('CREDIT_TERM',
                                                    app_train['AMT_ANNUITY'] / app_train[
                                                        'AMT_CREDIT'])
        app_train_poly = app_train_poly.add_columns('DAYS_EMPLOYED_PERCENT',
                                                    app_train['DAYS_EMPLOYED'] / app_train[
                                                        'DAYS_BIRTH'])

        app_test_poly = app_test_poly.add_columns('CREDIT_INCOME_PERCENT',
                                                  app_test['AMT_CREDIT'] / app_test[
                                                      'AMT_INCOME_TOTAL'])
        app_test_poly = app_test_poly.add_columns('ANNUITY_INCOME_PERCENT',
                                                  app_test['AMT_ANNUITY'] / app_test[
                                                      'AMT_INCOME_TOTAL'])
        app_test_poly = app_test_poly.add_columns('CREDIT_TERM',
                                                  app_test['AMT_ANNUITY'] / app_test['AMT_CREDIT'])
        app_test_poly = app_test_poly.add_columns('DAYS_EMPLOYED_PERCENT',
                                                  app_test['DAYS_EMPLOYED'] / app_test['DAYS_BIRTH'])

        # Align the dataframes
        train_columns = app_train_poly.data(verbose).columns
        test_columns = app_test_poly.data(verbose).columns
        for c in train_columns:
            if c not in test_columns:
                app_train_poly = app_train_poly.drop(c)

        app_train_poly.isnull().sum().data(verbose=verbose)

        print(app_train_poly.shape().data(verbose=verbose), app_test_poly.shape().data(verbose=verbose))

        # Drop the target from the training data
        col_2 = app_train.data(verbose).columns
        if 'TARGET' in col_2:
            train = app_train.drop(columns=['TARGET'])
        else:
            train = app_train.copy()

        # Feature names
        features = list(train.data(verbose).columns)

        # Copy of the testing data
        test = app_test.copy()

        # Median imputation of missing values
        imputer = Imputer(strategy='median')

        # Fit on the training data
        imputer.fit(train)

        # Transform both training and testing data
        train = imputer.transform(train)
        test = imputer.transform(test)

        print('Training data shape: ', train.shape().data(verbose))
        print('Testing data shape: ', test.shape().data(verbose))

        train = train.add_columns('TARGET', train_labels)
        train.head().data(verbose=verbose)

        MAX_EVALS = 5

        features = train.sample(n=16000, random_state=42)
        print(features.shape)
        labels = features['TARGET']
        features = features.drop(columns=['TARGET', 'SK_ID_CURR'])
        test = test.drop(columns=['SK_ID_CURR'])  # manually added

        model = LGBMClassifier()
        default_params = model.get_params()
        model.fit(features, labels)
        score = model.score(test,
                            test_labels['TARGET'],
                            score_type='auc').data(verbose=verbose)
        print('The model scores {} ROC AUC on the test set.'.format(score))

        def objective(hyperparameters, iteration):

            if 'n_estimators' in hyperparameters.keys():
                del hyperparameters['n_estimators']
            model = LGBMClassifier(**hyperparameters)
            model.fit(features, labels)
            score = model.score(test,
                                test_labels['TARGET'],
                                score_type='auc').data(verbose=verbose)
            return [score, hyperparameters, iteration]

        score, params, iteration = objective(default_params, 1)

        print('The cross-validation ROC AUC was {}.'.format(score))

        param_grid = {
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'num_leaves': list(range(20, 150)),
            'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
            'subsample_for_bin': list(range(20000, 300000, 20000)),
            'min_child_samples': list(range(20, 500, 5)),
            'reg_alpha': list(np.linspace(0, 1)),
            'reg_lambda': list(np.linspace(0, 1)),
            'colsample_bytree': list(np.linspace(0.6, 1, 10)),
            'subsample': list(np.linspace(0.5, 1, 100)),
            'is_unbalance': [True, False]
        }

        import random

        random.seed(50)

        # Randomly sample a boosting type
        boosting_type = random.sample(param_grid['boosting_type'], 1)[0]

        # Set subsample depending on boosting type
        subsample = 1.0 if boosting_type == 'goss' else random.sample(param_grid['subsample'], 1)[0]

        print('Boosting type: ', boosting_type)
        print('Subsample ratio: ', subsample)

        def grid_search(param_grid, max_evals=MAX_EVALS):
            """Grid search algorithm (with limit on max evals)"""
            results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                                   index=list(range(MAX_EVALS)))
            keys, values = zip(*param_grid.items())
            i = 0
            for v in itertools.product(*values):
                hyperparameters = dict(zip(keys, v))
                hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                    'subsample']
                eval_results = objective(hyperparameters, i)
                results.loc[i, :] = eval_results
                i += 1
                if i > max_evals:
                    break
            results.sort_values('score', ascending=False, inplace=True)
            results.reset_index(inplace=True)
            return results

        grid_results = grid_search(param_grid)
        print('The best validation score was {:.5f}'.format(grid_results.loc[0, 'score']))
        print('\nThe best hyperparameters were:')
        import pprint
        pprint.pprint(grid_results.loc[0, 'params'])

        grid_search_params = grid_results.loc[0, 'params']
        grid_search_params['random_state'] = 42

        model = LGBMClassifier(**grid_search_params)
        model.fit(features, labels)

        score = model.score(test,
                            test_labels['TARGET'],
                            score_type='auc').data(verbose=verbose)

        print('The best model from grid search scores {} ROC AUC on the test set.'.format(score))

        random.seed(50)

        # Randomly sample from dictionary
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        # Deal with subsample ratio
        random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

        random_params

        # Function to calculate missing values by column# Funct
        def random_search(param_grid, max_evals=MAX_EVALS):
            """Random search for hyperparameter optimization"""

            # Dataframe for results
            results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                                   index=list(range(MAX_EVALS)))

            # Keep searching until reach max evaluations
            for i in range(max_evals):
                # Choose random hyperparameters
                hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
                hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                    'subsample']

                # Evaluate randomly selected hyperparameters
                eval_results = objective(hyperparameters, i)

                results.loc[i, :] = eval_results

            # Sort with best score on top
            results.sort_values('score', ascending=False, inplace=True)
            results.reset_index(inplace=True)
            return results

        random_results = random_search(param_grid)

        print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
        print('\nThe best hyperparameters were:')

        import pprint
        pprint.pprint(random_results.loc[0, 'params'])

        random_search_params = random_results.loc[0, 'params']

        # Create, train, test model
        random_search_params['random_state'] = 42
        model = LGBMClassifier(**random_search_params)
        labels = app_train['TARGET']
        features = app_train_poly.drop(columns=['SK_ID_CURR'])
        model.fit(app_train_poly, labels)

        score = model.score(app_test_poly,
                            test_labels['TARGET'],
                            score_type='auc').data(verbose=verbose)
        print('The best model from random search scores {} ROC AUC on the test set.'.format(score))

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

    workload = fork_taozhongxiao_start_here_a_gentle_introduction()

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

    executor.end_to_end_run(workload=workload, root_data=root_data, verbose=0)

    # executor.store_experiment_graph(database_path)
    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    print('finished execution in {} seconds'.format(elapsed))
