#!/usr/bin/env python

"""Optimized workload 1 for Home Credit Default Risk Competition
    This is the optimized version of the baseline workload_1 script.

   For now, I removed the Kfold and Gradient Boosted Tree models
   TODO: Add Kfold and Gradient Boosted Tree
"""
import os
import warnings

# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
# numpy and pandas for data manipulation
import pandas as pd
import seaborn as sns

# Experiment Graph

# Suppress warnings
warnings.filterwarnings('ignore')


def run(execution_environment, root_data):
    print(os.listdir(root_data))
    app_train = execution_environment.load(root_data + '/home-credit-default-risk/application_train.csv')
    print('Training data shape: ', app_train.shape().data())
    app_train.head().data()

    app_test = execution_environment.load(root_data + '/home-credit-default-risk/application_test.csv')
    print('Testing data shape: ', app_test.shape().data())
    app_test.head().data()

    app_train['TARGET'].value_counts().data()

    app_train['TARGET'].data().astype(int).plot.hist()

    # Function to calculate missing values by column# Funct
    def missing_values_table(dataset):
        # Total missing values
        mis_val = dataset.isnull().sum().data()

        mis_val_percent = 100 * mis_val / len(dataset.data())

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
        print("Your selected dataframe has " + str(dataset.shape().data()[1]) + " columns.\n"
                                                                                "There are " + str(
            mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    missing_values = missing_values_table(app_train)
    missing_values.head(20)

    app_train.dtypes().data().value_counts()

    app_train.select_dtypes('object').nunique().data()

    from sklearn_helper.preprocessing import LabelEncoder
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    for col in app_train.select_dtypes('object').data().columns:
        # we are not using nunique because it discard nan
        if app_train[col].nunique(dropna=False).data() <= 2:
            le.fit(app_train[col])
            app_train = app_train.replace_columns(col, le.transform(app_train[col]))
            app_test = app_test.replace_columns(col, le.transform(app_test[col]))

            # Keep track of how many columns were label encoded
            le_count += 1
    print('%d columns were label encoded.' % le_count)

    app_train = app_train.onehot_encode()
    app_test = app_test.onehot_encode()

    print('Training Features shape: ', app_train.shape().data())
    print('Testing Features shape: ', app_test.shape().data())

    train_labels = app_train['TARGET']
    train_columns = app_train.data().columns
    test_columns = app_test.data().columns
    for c in train_columns:
        if c not in test_columns:
            app_train = app_train.drop(c)

    app_train = app_train.add_columns('TARGET', train_labels)

    print('Training Features shape: ', app_train.shape().data())
    print('Testing Features shape: ', app_test.shape().data())

    (app_train['DAYS_BIRTH'] / 365).describe().data()

    app_train['DAYS_EMPLOYED'].describe().data()

    app_train['DAYS_EMPLOYED'].data().plot.hist(title='Days Employment Histogram')
    plt.xlabel('Days Employment')

    anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
    non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
    print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean().data()))
    print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean().data()))
    print('There are %d anomalous days of employment' % anom.shape().data()[0])

    days_employed_anom = app_train["DAYS_EMPLOYED"] == 365243
    app_train = app_train.add_columns('DAYS_EMPLOYED_ANOM', days_employed_anom)
    temp = app_train['DAYS_EMPLOYED'].replace({365243: np.nan})
    app_train = app_train.drop('DAYS_EMPLOYED')
    app_train = app_train.add_columns('DAYS_EMPLOYED', temp)

    app_train["DAYS_EMPLOYED"].data().plot.hist(title='Days Employment Histogram');
    plt.xlabel('Days Employment')

    days_employed_anom = app_test["DAYS_EMPLOYED"] == 365243
    app_test = app_test.add_columns('DAYS_EMPLOYED_ANOM', days_employed_anom)
    temp = app_test['DAYS_EMPLOYED'].replace({365243: np.nan})
    app_test = app_test.drop('DAYS_EMPLOYED')
    app_test = app_test.add_columns('DAYS_EMPLOYED', temp)
    print('There are %d anomalies in the test data out of %d entries'
          % (app_test['DAYS_EMPLOYED_ANOM'].sum().data(),
             app_test.shape().data()[0]))

    correlations = app_train.corr().data()
    top = correlations['TARGET'].sort_values()
    # Display correlations
    print('Most Positive Correlations:\n', top.tail(15))
    print('\nMost Negative Correlations:\n', top.head(15))

    abs_age = app_train['DAYS_BIRTH'].abs()
    app_train = app_train.drop('DAYS_BIRTH')
    app_train = app_train.add_columns('DAYS_BIRTH', abs_age)
    app_train['DAYS_BIRTH'].corr(app_train['TARGET']).data()

    # Set the style of plots
    plt.style.use('fivethirtyeight')

    # Plot the distribution of ages in years
    plt.hist((app_train['DAYS_BIRTH'] / 365).data(), edgecolor='k', bins=25)
    plt.title('Age of Client')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')

    plt.figure(figsize=(10, 8))
    # KDE plot of loans that were repaid on time
    sns.kdeplot((app_train[app_train['TARGET'] == 0]['DAYS_BIRTH'] / 365).data(), label='target == 0')
    # KDE plot of loans which were not repaid on time
    sns.kdeplot((app_train[app_train['TARGET'] == 1]['DAYS_BIRTH'] / 365).data(), label='target == 1')
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
    age_data.head(10).data()

    age_groups = age_data.groupby('YEARS_BINNED').mean()
    age_groups.data()

    plt.figure(figsize=(8, 8))

    # Graph the age bins and the average of the target as a bar plot
    plt.bar(age_groups.data().index.astype(str), age_groups.data()['TARGET'] * 100)

    # Plot labeling
    plt.xticks(rotation=75)
    plt.xlabel('Age Group (years)')
    plt.ylabel('Failure to Repay (%)')
    plt.title('Failure to Repay by Age Group')

    ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    ext_data_corrs = ext_data.corr().data()

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
        sns.kdeplot(source_data[app_train[column].notna()][column].data(), label='target == 0')
        # plot loans that were not repaid
        source_data = app_train[[column, 'TARGET']][app_train['TARGET'] == 1]
        sns.kdeplot(source_data[app_train[column].notna()][column].data(), label='target == 1')

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
    grid = sns.PairGrid(data=plot_data.data(), size=3, diag_sharey=False,
                        hue='TARGET',
                        vars=[x for x in list(plot_data.data().columns) if x != 'TARGET'])

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
    from sklearn_helper.preprocessing import Imputer

    imputer = Imputer(strategy='median')

    poly_target = poly_features['TARGET']

    poly_features = poly_features.drop(columns=['TARGET'])

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    from sklearn_helper.preprocessing import PolynomialFeatures

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    print('Polynomial Features shape: ', poly_features.shape().data())

    new_names = poly_transformer.get_feature_names(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                   'EXT_SOURCE_3', 'DAYS_BIRTH'])

    poly_features = poly_features.set_columns(new_names)

    # Add in the target
    poly_features = poly_features.add_columns('TARGET', poly_target)

    # Find the correlations with the target
    poly_corrs = poly_features.corr().data()['TARGET'].sort_values()

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
    train_columns = app_train_poly.data().columns
    test_columns = app_test_poly.data().columns
    for c in train_columns:
        if c not in test_columns:
            app_train_poly = app_train_poly.drop(c)

    # Print out the new shapes
    print('Training data with polynomial features shape: ',
          app_train_poly.shape().data())
    print('Testing data with polynomial features shape:  ',
          app_test_poly.shape().data())

    app_train_domain = app_train.copy()
    app_test_domain = app_test.copy()

    app_train_domain = app_train_domain.add_columns('CREDIT_INCOME_PERCENT',
                                                    app_train_domain['AMT_CREDIT'] / app_train_domain[
                                                        'AMT_INCOME_TOTAL'])
    app_train_domain = app_train_domain.add_columns('ANNUITY_INCOME_PERCENT',
                                                    app_train_domain['AMT_ANNUITY'] / app_train_domain[
                                                        'AMT_INCOME_TOTAL'])
    app_train_domain = app_train_domain.add_columns('CREDIT_TERM',
                                                    app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT'])
    app_train_domain = app_train_domain.add_columns('DAYS_EMPLOYED_PERCENT',
                                                    app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH'])

    app_test_domain = app_test_domain.add_columns('CREDIT_INCOME_PERCENT',
                                                  app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL'])
    app_test_domain = app_test_domain.add_columns('ANNUITY_INCOME_PERCENT',
                                                  app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL'])
    app_test_domain = app_test_domain.add_columns('CREDIT_TERM',
                                                  app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT'])
    app_test_domain = app_test_domain.add_columns('DAYS_EMPLOYED_PERCENT',
                                                  app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH'])

    plt.figure(figsize=(12, 20))
    # iterate through the new features
    for i, column in enumerate([
        'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM',
        'DAYS_EMPLOYED_PERCENT'
    ]):
        # create a new subplot for each source
        plt.subplot(4, 1, i + 1)
        # plot repaid loans
        negative = app_train_domain[[column, 'TARGET']][app_train['TARGET'] == 0]
        sns.kdeplot(
            negative[app_train_domain[column].notna()][column].data(),
            label='target == 0')
        # plot loans that were not repaid
        positive = app_train_domain[[column, 'TARGET']][app_train['TARGET'] == 1]
        sns.kdeplot(
            positive[app_train_domain[column].notna()][column].data(),
            label='target == 1')

        # Label the plots
        plt.title('Distribution of %s by Target Value' % column)
        plt.xlabel('%s' % column)
        plt.ylabel('Density')

    plt.tight_layout(h_pad=2.5)

    from sklearn_helper.preprocessing import MinMaxScaler, Imputer

    # Drop the target from the training data
    columns = app_train.data().columns
    if 'TARGET' in columns:
        train = app_train.drop(columns=['TARGET'])
    else:
        train = app_train.copy()

    # Feature names
    features = list(train.data().columns)

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

    print('Training data shape: ', train.shape().data())
    print('Testing data shape: ', test.shape().data())

    from sklearn_helper.linear_model import LogisticRegression

    # Make the model with the specified regularization parameter
    log_reg = LogisticRegression(C=0.0001)

    # Train on the training data
    log_reg.fit(train, train_labels)

    # Make predictions
    # Make sure to select the second column only
    log_reg_pred = log_reg.predict_proba(test)[1]

    # Submission data
    log_reg_pred.setname('TARGET')
    submit = app_test['SK_ID_CURR'].concat(log_reg_pred)
    submit.head().data()

    from sklearn_helper.ensemble import RandomForestClassifier

    # Make the random forest classifier
    random_forest = RandomForestClassifier(n_estimators=10, random_state=50, verbose=1, n_jobs=-1)

    # Train on the training data
    random_forest.fit(train, train_labels)

    # Extract feature importance
    feature_importances = random_forest.feature_importances(features)

    # Make predictions on the test data
    predictions = random_forest.predict_proba(test)[1]

    # Score = 0.678
    # Submission dataframe
    predictions.setname('TARGET')
    submit = app_test['SK_ID_CURR'].concat(predictions)
    submit.head().data()

    poly_features_names = list(app_train_poly.data().columns)

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
    predictions = random_forest_poly.predict_proba(poly_features_test)[1]

    # Score = 0.678
    # Submission dataframe
    predictions.setname('TARGET')
    submit = app_test['SK_ID_CURR'].concat(predictions)
    submit.head().data()

    app_train_domain = app_train_domain.drop(columns='TARGET')

    domain_features_names = list(app_train_domain.data().columns)

    # Impute the domainnomial features
    imputer = Imputer(strategy='median')

    domain_features = imputer.fit_transform(app_train_domain)
    domain_features_test = imputer.transform(app_test_domain)

    # Scale the domainnomial features
    scaler = MinMaxScaler(feature_range=(0, 1))

    domain_features = scaler.fit_transform(domain_features)
    domain_features_test = scaler.transform(domain_features_test)

    # Train on the training data
    random_forest_domain = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
    random_forest_domain.fit(domain_features, train_labels)

    # Extract feature importances
    feature_importances_domain = random_forest_domain.feature_importances(domain_features_names)

    # Make predictions on the test data
    predictions = random_forest_domain.predict_proba(domain_features_test)[1]

    # Score = 0.678
    # Make a submission dataframe
    predictions.setname('TARGET')
    submit = app_test['SK_ID_CURR'].concat(predictions)
    submit.head().data()

    def plot_feature_importances(df):
        """
        Plot importances returned by a model. This can work with any measure of
        feature importance provided that higher importance is better.

        Args:
            df (dataframe): feature importances. Must have the features in a column
            called `features` and the importances in a column called `importance

        Returns:
            shows a plot of the 15 most importance features

            df (dataframe): feature importances sorted by importance (highest to lowest)
            with a column for normalized importance
            """

        # Sort features according to importance
        df = df.sort_values('importance', ascending=False)

        # Normalize the feature importances to add up to one
        df = df.add_columns('importance_normalized', df['importance'] / df['importance'].sum().data())

        # Make a horizontal bar chart of feature importances
        plt.figure(figsize=(10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        ax.barh(list(reversed(list(df.data().index[:15]))),
                df['importance_normalized'].data().head(15),
                align='center', edgecolor='k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(df.data().index[:15]))))
        ax.set_yticklabels(df['feature'].data().head(15))

        # Plot labeling
        plt.xlabel('Normalized Importance')
        plt.title('Feature Importances')
        plt.show()

        return df

    # Show the feature importances for the default features
    feature_importances_sorted = plot_feature_importances(feature_importances)

    feature_importances_domain_sorted = plot_feature_importances(feature_importances_domain)

    from sklearn_helper.sklearn_wrappers import LGBMClassifier

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

            features_columns = lgb_featres.data().columns
            test_features_columns = test_features.data().columns
            for c in features_columns:
                if c not in test_features_columns:
                    lgb_featres = lgb_featres.drop(c)

            # No categorical indices to record
            cat_indices = 'auto'

        # # Integer label encoding
        # elif encoding == 'le':
        #
        #     # Create a label encoder
        #     label_encoder = LabelEncoder()
        #
        #     # List for storing categorical indices
        #     cat_indices = []
        #
        #     # Iterate through each column
        #     for i, col in enumerate(lgb_featres):
        #         if lgb_featres[col].dtype == 'object':
        #             # Map the categorical features to integers
        #             lgb_featres[col] = label_encoder.fit_transform(np.array(lgb_featres[col].astype(str)).reshape((-1,)))
        #             test_features[col] = label_encoder.transform(
        #                 np.array(test_features[col].astype(str)).reshape((-1,)))
        #
        #             # Record the categorical indices
        #             cat_indices.append(i)

        # Catch error if label encoding scheme is not valid
        else:
            raise ValueError("Encoding must be either 'ohe' or 'le'")

        print('Training Data Shape: ', lgb_featres.shape().data())
        print('Testing Data Shape: ', test_features.shape().data())

        # Extract feature names
        feature_names = list(lgb_featres.data().columns)

        # Create the model
        model = LGBMClassifier(n_estimators=10, objective='binary',
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
        test_predictions = model.predict_proba(test_features, custom_args={'num_iteration': best_iteration})[1]

        test_predictions = test_predictions.setname('TARGET')
        # Make the submission dataframe
        submission = test_ids.concat(test_predictions)

        feature_importances = model.feature_importances(feature_names)

        return feature_importances

    fi = model(app_train, app_test)
    fi_sorted = plot_feature_importances(fi)

    app_train_domain = app_train_domain.add_columns('TARGET', train_labels)

    # Test the domain knowledge features
    fi_domain = model(app_train_domain, app_test_domain)
    fi_sorted = plot_feature_importances(fi_domain)

    # the total time is captured by the profiler,
    # here we return the experiment_graphs load, save, and the total time the system spent in training models
    return execution_environment.time_manager.get('model-training', 0)


from experiment_graph.execution_environment import ExecutionEnvironment

ee = ExecutionEnvironment('dedup')
execution_start = datetime.now()
ROOT_PACKAGE_DIRECTORY = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/jupyter'
root_data = ROOT_PACKAGE_DIRECTORY + '/data'
DATABASE_PATH = root_data + '/experiment_graphs/home-credit-default-risk/environment_dedup'
# ee.load_environment(DATABASE_PATH)
run(ee, root_data)
ee.save_history(DATABASE_PATH)

execution_end = datetime.now()
elapsed = (execution_end - execution_start).total_seconds()

print('finished execution in {} seconds'.format(elapsed))
