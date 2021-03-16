#!/usr/bin/env python

"""Optimized Fork of Workload 1

This script is the optimized version of the fork of workload 1 submitted by user taozhongxiao
which utilizes our Experiment Graph for optimizing the workload.
"""
import os
import warnings
# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib
import pandas as pd

from Reuse import LinearTimeReuse
from data_storage import DedupedStorageManager
from execution_environment import ExecutionEnvironment
from executor import CollaborativeExecutor
from materialization_methods import StorageAwareMaterializer

matplotlib.use('ps')
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

        print('LGBMClassifier with AUC score: {}'.format(score))

        def objective(hyperparameters, iteration):
            if 'n_estimators' in hyperparameters.keys():
                del hyperparameters['n_estimators']
            model = LGBMClassifier(**hyperparameters)
            model.fit(train_features, train_labels)
            score = model.score(test_features,
                                test_labels['TARGET'],
                                score_type='auc').data(verbose=verbose)
            return [score['auc'], hyperparameters, iteration]

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
            print(results)
            results.sort_values('score', ascending=False, inplace=True)
            results.reset_index(inplace=True)
            return results

        grid_results = grid_search(param_grid)
        print('The best validation score was {}'.format(grid_results.loc[0, 'score']))
        print('\nThe best hyperparameters were:')
        import pprint
        pprint.pprint(grid_results.loc[0, 'params'])

        grid_search_params = grid_results.loc[0, 'params']
        grid_search_params['random_state'] = 42
        model = LGBMClassifier(**grid_search_params)
        model.fit(train_features, train_labels)

        score = model.score(test_features,
                            test_labels['TARGET'],
                            score_type='auc').data(verbose=verbose)

        print
        'The best model from grid search scores {} ROC AUC on the test set.'.format(score)

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

        print('The best model from random search scores {} ROC AUC on the test set.'.format(score))

        return True


if __name__ == "__main__":
    ROOT = '/Users/bede01/Documents/work/phd-papers/published/ml-workload-optimization'
    root_data = ROOT + '/data'

    workload = fork_taozhongxiao_start_here_a_gentle_introduction()

    mat_budget = 16.0 * 1024.0 * 1024.0
    sa_materializer = StorageAwareMaterializer(storage_budget=mat_budget)

    ee = ExecutionEnvironment(DedupedStorageManager(), reuse_type=LinearTimeReuse.NAME)


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
