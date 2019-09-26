#!/usr/bin/env python

"""Baseline workload 1 for Home Credit Default Risk Competition
   The code here, is the original code posted as a notebook for the Kaggle competitiong.
   The notebook can be found here: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

   For now, I removed the Kfold and Gradient Boosted Tree models
   TODO: Add Kfold and Gradient Boosted Tree
"""
import os
# File system management
# Suppress warnings
import warnings

# plotting libraries
from datetime import datetime
import matplotlib

from experiment_graph.executor import BaselineExecutor
from experiment_graph.workload import Workload

matplotlib.use('ps')
import matplotlib.pyplot as plt
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd
import seaborn as sns
# sklearn preprocessing for dealing with categorical variables
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


class cridata_start_here_a_gentle_introduction(Workload):

    def run(self, root_data):
        print(os.listdir(root_data))
        app_train = pd.read_csv(root_data + '/kaggle_home_credit/application_train.csv')
        print('Training data shape: ', app_train.shape)
        app_train.head()

        app_test = pd.read_csv(root_data + '/kaggle_home_credit/application_test.csv')
        print('Testing data shape: ', app_test.shape)
        app_test.head()

        app_train['TARGET'].value_counts()

        app_train['TARGET'].astype(int).plot.hist()

        test_labels = pd.read_csv(root_data + '/kaggle_home_credit/application_test_labels.csv')

        # Function to calculate missing values by column# Funct
        def missing_values_table(df):
            # Total missing values
            mis_val = df.isnull().sum()

            # Percentage of missing values
            mis_val_percent = 100 * df.isnull().sum() / len(df)

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
            print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                      "There are " + str(
                mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")

            # Return the dataframe with missing information
            return mis_val_table_ren_columns

        missing_values = missing_values_table(app_train)
        missing_values.head(20)

        app_train.dtypes.value_counts()

        app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

        # Create a label encoder object
        le = LabelEncoder()
        le_count = 0

        # Iterate through the columns
        for col in app_train:
            if app_train[col].dtype == 'object':
                # If 2 or fewer unique categories
                if len(list(app_train[col].unique())) <= 2:
                    # Train on the training data
                    le.fit(app_train[col])
                    # Transform both training and testing data
                    app_train[col] = le.transform(app_train[col])
                    app_test[col] = le.transform(app_test[col])

                    # Keep track of how many columns were label encoded
                    le_count += 1

        print('%d columns were label encoded.' % le_count)

        # one-hot encoding of categorical variables
        app_train = pd.get_dummies(app_train)
        app_test = pd.get_dummies(app_test)

        print('Training Features shape: ', app_train.shape)
        print('Testing Features shape: ', app_test.shape)

        train_labels = app_train['TARGET']

        # Align the training and testing data, keep only columns present in both dataframes
        app_train, app_test = app_train.align(app_test, join='inner', axis=1)

        print('Training Features shape: ', app_train.shape)
        print('Testing Features shape: ', app_test.shape)

        # Add the target back in
        app_train['TARGET'] = train_labels

        # CHANGE
        # (app_train['DAYS_BIRTH'] / -365).describe()
        #
        # app_train['DAYS_EMPLOYED'].describe()
        #
        # app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
        # plt.xlabel('Days Employment')
        #
        # anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
        # non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
        # print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
        # print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
        # print('There are %d anomalous days of employment' % len(anom))
        #
        # # Create an anomalous flag column
        # app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
        #
        # # Replace the anomalous values with nan
        # app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
        #
        # app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
        # plt.xlabel('Days Employment')
        #
        # app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
        # app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
        # print('There are %d anomalies in the test data out of %d entries' % (
        #     app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

        # Find correlations with the target and sort
        correlations = app_train.corr()['TARGET'].sort_values()

        # Display correlations
        print('Most Positive Correlations:\n', correlations.tail(15))
        print('\nMost Negative Correlations:\n', correlations.head(15))

        # Find the correlation of the positive days since birth and target
        app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
        app_train['DAYS_BIRTH'].corr(app_train['TARGET'])

        # Set the style of plots
        plt.style.use('fivethirtyeight')

        # Plot the distribution of ages in years
        plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
        plt.title('Age of Client')
        plt.xlabel('Age (years)')
        plt.ylabel('Count')

        plt.figure(figsize=(10, 8))

        # KDE plot of loans that were repaid on time
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label='target == 0')

        # KDE plot of loans which were not repaid on time
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label='target == 1')

        # Labeling of plot
        plt.xlabel('Age (years)')
        plt.ylabel('Density')
        plt.title('Distribution of Ages')

        # Age information into a separate dataframe
        age_data = app_train[['TARGET', 'DAYS_BIRTH']]
        age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

        # Bin the age data
        age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))
        age_data.head(10)

        # Group by the bin and calculate averages
        age_groups = age_data.groupby('YEARS_BINNED').mean()
        age_groups

        plt.figure(figsize=(8, 8))

        # Graph the age bins and the average of the target as a bar plot
        plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

        # Plot labeling
        plt.xticks(rotation=75)
        plt.xlabel('Age Group (years)')
        plt.ylabel('Failure to Repay (%)')
        plt.title('Failure to Repay by Age Group')

        # Extract the EXT_SOURCE variables and show correlations
        ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
        ext_data_corrs = ext_data.corr()
        ext_data_corrs

        plt.figure(figsize=(8, 6))

        # Heatmap of correlations
        sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
        plt.title('Correlation Heatmap')

        plt.figure(figsize=(10, 12))

        # iterate through the sources
        for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
            # create a new subplot for each source
            plt.subplot(3, 1, i + 1)
            # plot repaid loans
            sns.kdeplot(app_train[source][(app_train[source].notna()) & (app_train['TARGET'] == 0)],
                        label='target == 0')
            # plot loans that were not repaid
            sns.kdeplot(app_train[source][(app_train[source].notna()) & (app_train['TARGET'] == 1)],
                        label='target == 1')

            # Label the plots
            plt.title('Distribution of %s by Target Value' % source)
            plt.xlabel('%s' % source);
            plt.ylabel('Density');

        plt.tight_layout(h_pad=2.5)

        # Copy the data for plotting
        plot_data = ext_data.drop(columns=['DAYS_BIRTH']).copy()

        # Add in the age of the client in years
        plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

        # Drop na values and limit to first 100000 rows
        plot_data = plot_data.dropna().loc[:100000, :]

        # Create the pairgrid object
        grid = sns.PairGrid(data=plot_data, size=3, diag_sharey=False,
                            hue='TARGET',
                            vars=[x for x in list(plot_data.columns) if x != 'TARGET'])

        # Upper is a scatter plot
        grid.map_upper(plt.scatter, alpha=0.2)

        # Diagonal is a histogram
        grid.map_diag(sns.kdeplot)

        # Bottom is density plot
        grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r)

        plt.suptitle('Ext Source and Age Features Pairs Plot', size=32, y=1.05)

        # Make a new dataframe for polynomial features
        poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
        poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

        # imputer for handling missing values
        from sklearn.preprocessing import Imputer

        imputer = Imputer(strategy='median')

        poly_target = poly_features['TARGET']

        poly_features = poly_features.drop(columns=['TARGET'])

        # Need to impute missing values
        poly_features = imputer.fit_transform(poly_features)
        poly_features_test = imputer.transform(poly_features_test)

        from sklearn.preprocessing import PolynomialFeatures

        # Create the polynomial object with specified degree
        poly_transformer = PolynomialFeatures(degree=3)

        # Train the polynomial features
        poly_transformer.fit(poly_features)

        # Transform the features
        poly_features = poly_transformer.transform(poly_features)
        poly_features_test = poly_transformer.transform(poly_features_test)
        print('Polynomial Features shape: ', poly_features.shape)

        poly_transformer.get_feature_names(
            input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])

        # Create a dataframe of the features
        poly_features = pd.DataFrame(poly_features,
                                     columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                 'EXT_SOURCE_3', 'DAYS_BIRTH']))

        # Add in the target
        poly_features['TARGET'] = poly_target

        # Find the correlations with the target
        poly_corrs = poly_features.corr()['TARGET'].sort_values()

        # Display most negative and most positive
        print(poly_corrs.head(10))
        print(poly_corrs.tail(5))

        # Put test features into dataframe
        poly_features_test = pd.DataFrame(poly_features_test,
                                          columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                      'EXT_SOURCE_3', 'DAYS_BIRTH']))

        # Merge polynomial features into training dataframe
        poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
        app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')

        # Merge polnomial features into testing dataframe
        poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
        app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

        # Align the dataframes
        app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

        # Print out the new shapes
        print('Training data with polynomial features shape: ', app_train_poly.shape)
        print('Testing data with polynomial features shape:  ', app_test_poly.shape)

        # CHANGE
        # app_train_domain = app_train.copy()
        # app_test_domain = app_test.copy()
        #
        # app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain[
        #     'AMT_INCOME_TOTAL']
        # app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain[
        #     'AMT_INCOME_TOTAL']
        # app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
        # app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
        #
        # app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
        # app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
        # app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
        # app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']

        from sklearn.preprocessing import MinMaxScaler, Imputer

        # Drop the target from the training data
        if 'TARGET' in app_train:
            train = app_train.drop(columns=['TARGET'])
        else:
            train = app_train.copy()

        # Feature names
        features = list(train.columns)

        # Copy of the testing data
        test = app_test.copy()

        # Median imputation of missing values
        imputer = Imputer(strategy='median')

        # Scale each feature to 0-1
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit on the training data
        imputer.fit(train)

        # Transform both training and testing data
        train = imputer.transform(train)
        test = imputer.transform(test)

        # Repeat with the scaler
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

        print('Training data shape: ', train.shape)
        print('Testing data shape: ', test.shape)

        from sklearn.linear_model import LogisticRegression

        # Make the model with the specified regularization parameter
        log_reg = LogisticRegression(C=0.0001)

        # Train on the training data
        log_reg.fit(train, train_labels)

        # Make predictions
        # Make sure to select the second column only
        log_reg_pred = log_reg.predict_proba(test)[:, 1]
        score = roc_auc_score(test_labels['TARGET'], log_reg_pred)
        print 'Logistic Regression with AUC score: {}'.format(score)

        from sklearn.ensemble import RandomForestClassifier

        # Make the random forest classifier
        random_forest = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)

        # Train on the training data
        random_forest.fit(train, train_labels)

        # CHANGE
        # # Extract feature importances
        # feature_importance_values = random_forest.feature_importances_
        # feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

        # Make predictions on the test data
        predictions = random_forest.predict_proba(test)[:, 1]

        # Score = 0.678
        # Make a submission dataframe
        score = roc_auc_score(test_labels['TARGET'], predictions)
        print 'Random Forest Simple Data with AUC score: {}'.format(score)

        poly_features_names = list(app_train_poly.columns)

        # Impute the polynomial features
        imputer = Imputer(strategy='median')

        poly_features = imputer.fit_transform(app_train_poly)
        poly_features_test = imputer.transform(app_test_poly)

        # Scale the polynomial features
        scaler = MinMaxScaler(feature_range=(0, 1))

        poly_features = scaler.fit_transform(poly_features)
        poly_features_test = scaler.transform(poly_features_test)

        random_forest_poly = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
        random_forest_poly.fit(poly_features, train_labels)

        # Make predictions on the test data
        predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]

        # Score = 0.678
        # Make a submission dataframe
        score = roc_auc_score(test_labels['TARGET'], predictions)
        print 'Random Forest Poly Data with AUC score: {}'.format(score)

        # CHANGE
        # app_train_domain = app_train_domain.drop(columns='TARGET')
        #
        # domain_features_names = list(app_train_domain.columns)
        #
        # # Impute the domainnomial features
        # imputer = Imputer(strategy='median')
        #
        # domain_features = imputer.fit_transform(app_train_domain)
        # domain_features_test = imputer.transform(app_test_domain)
        #
        # # Scale the domainnomial features
        # scaler = MinMaxScaler(feature_range=(0, 1))
        #
        # domain_features = scaler.fit_transform(domain_features)
        # domain_features_test = scaler.transform(domain_features_test)
        #
        # # Train on the training data
        # random_forest_domain = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
        # random_forest_domain.fit(domain_features, train_labels)
        #
        # # Extract feature importances
        # feature_importance_values_domain = random_forest_domain.feature_importances_
        # feature_importances_domain = pd.DataFrame(
        #     {'feature': domain_features_names, 'importance': feature_importance_values_domain})
        #
        # # Make predictions on the test data
        # predictions = random_forest_domain.predict_proba(domain_features_test)[:, 1]
        #
        # # Score = 0.679
        # # Make a submission dataframe
        # score = roc_auc_score(test_labels['TARGET'], predictions)
        # print 'Random Forest Domain Data with AUC score: {}'.format(score)

        # CHANGE entire function
        def show_feature_importances(model, features):

            plt.figure(figsize=(12, 8))
            # Dataframe of feature importances sorted from most to least
            results = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
            results = results.sort_values('importance', ascending=False)

            # Display
            print(results.head(10))
            print('\nNumber of features with importance greater than 0.01 = ', np.sum(results['importance'] > 0.01))

            # Plot of feature importances as horizontal bar chart
            results.head(20).plot(x='feature', y='importance', kind='barh',
                                  color='red', edgecolor='k', title='Feature Importances');
            return results

        # Show the feature importances for the default features
        feature_importances = show_feature_importances(random_forest, features)

        # CHANGE
        # # Show the feature importances for the default features
        # feature_importances_sorted = plot_feature_importances(feature_importances)
        #
        # feature_importances_domain_sorted = plot_feature_importances(feature_importances_domain)

        import lightgbm as lgb

        # CHANGE
        # def model(features, test_features, encoding='ohe', n_folds=5):
        #
        #     """Train and test a light gradient boosting model using
        #     cross validation.
        #
        #     Parameters
        #     --------
        #         features (pd.DataFrame):
        #             dataframe of training features to use
        #             for training a model. Must include the TARGET column.
        #         test_features (pd.DataFrame):
        #             dataframe of testing features to use
        #             for making predictions with the model.
        #         encoding (str, default = 'ohe'):
        #             method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
        #             n_folds (int, default = 5): number of folds to use for cross validation
        #
        #     Return
        #     --------
        #         submission (pd.DataFrame):
        #             dataframe with `SK_ID_CURR` and `TARGET` probabilities
        #             predicted by the model.
        #         feature_importances (pd.DataFrame):
        #             dataframe with the feature importances from the model.
        #         valid_metrics (pd.DataFrame):
        #             dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        #
        #     """

        # Extract the ids
        train = np.array(app_train.drop(columns='TARGET'))
        test = np.array(app_test)

        # Extract the labels for training
        train_labels = np.array(train_labels).reshape((-1,))

        # Remove the ids and target
        # features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
        # test_features = test_features.drop(columns=['SK_ID_CURR'])

        # CHANGE
        # # One Hot Encoding
        # if encoding == 'ohe':
        #     features = pd.get_dummies(features)
        #     test_features = pd.get_dummies(test_features)
        #
        #     # Align the dataframes by the columns
        #     features, test_features = features.align(test_features, join='inner', axis=1)
        #
        #     # No categorical indices to record
        #     cat_indices = 'auto'
        #
        # # Integer label encoding
        # elif encoding == 'le':
        #
        #     # Create a label encoder
        #     label_encoder = LabelEncoder()
        #
        #     # List for storing categorical indices
        #     cat_indices = []

        # # Iterate through each column
        # for i, col in enumerate(features):
        #     if features[col].dtype == 'object':
        #         # Map the categorical features to integers
        #         features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
        #         test_features[col] = label_encoder.transform(
        #             np.array(test_features[col].astype(str)).reshape((-1,)))
        #
        #         # Record the categorical indices
        #         cat_indices.append(i)

        # # Catch error if label encoding scheme is not valid
        # else:
        #     raise ValueError("Encoding must be either 'ohe' or 'le'")

        # print('Training Data Shape: ', features.shape)
        # print('Testing Data Shape: ', test_features.shape)

        # Extract feature names
        # feature_names = list(features.columns)

        # Convert to np arrays
        # features = np.array(features)
        # test_features = np.array(test_features)

        # Create the model
        clf = lgb.LGBMClassifier(
            n_estimators=10000,
            learning_rate=0.1,
            subsample=.8,
            max_depth=7,
            reg_alpha=.1,
            reg_lambda=.1,
            min_split_gain=.01,
            min_child_weight=2
        )

        # Train the model
        clf.fit(train, train_labels, eval_metric='auc', verbose=False)

        # Record the best iteration
        best_iteration = clf.best_iteration_
        predictions = clf.predict_proba(test, num_iteration=best_iteration)[:, 1]
        score = roc_auc_score(test_labels['TARGET'], predictions)
        print 'LGBMClassifier with AUC score: {}'.format(score)
        # Record the feature importances
        # feature_importance_values = model.feature_importances_
        #
        # feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
        #
        # fi = model(app_train, app_test)
        # fi_sorted = plot_feature_importances(fi)
        #
        # app_train_domain['TARGET'] = train_labels
        #
        # # Test the domain knowledge features
        # fi_domain = model(app_train_domain, app_test_domain)
        # fi_sorted = plot_feature_importances(fi_domain)

        return True


if __name__ == "__main__":
    ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
    ROOT_PACKAGE = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer'
    root_data = ROOT + '/data'
    import sys

    sys.path.append(ROOT_PACKAGE)

    executor = BaselineExecutor()
    execution_start = datetime.now()
    workload = cridata_start_here_a_gentle_introduction()
    executor.end_to_end_run(workload=workload, root_data=root_data)

    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    print('finished execution in {} seconds'.format(elapsed))
