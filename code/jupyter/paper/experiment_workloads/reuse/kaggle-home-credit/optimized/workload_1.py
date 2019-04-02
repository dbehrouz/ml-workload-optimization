#!/usr/bin/env python

"""Optimized workload 1 for Home Credit Default Risk Competition
    This is the optimized version of the baseline workload_1 script.

   For now, I removed the Kfold and Gradient Boosted Tree models
   TODO: Add Kfold and Gradient Boosted Tree
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# Experiment Graph
from sklearn.preprocessing import LabelEncoder

# Add the experiment_graph package directory path
sys.path.append(sys.argv[1])
from experiment_graph.execution_environment import ExecutionEnvironment as ee

# numpy and pandas for data manipulation

# matplotlib and seaborn for plotting
# sklearn preprocessing for dealing with categorical variables

# Suppress warnings
warnings.filterwarnings('ignore')

ROOT_DIRECTORY = sys.argv[2]
GRAPH_LOCATION = sys.argv[3]
if os.path.isfile(GRAPH_LOCATION):
    print 'Load Existing Experiment Graph!!'
    ee.load_graph(GRAPH_LOCATION)
else:
    print 'No Experiment Graph Exists!!!'

print(os.listdir(ROOT_DIRECTORY))
app_train = ee.load(ROOT_DIRECTORY + '/home-credit-default-risk/application_train.csv')
print('Training data shape: ', app_train.shape().get())
app_train.head().get()

app_test = ee.load(ROOT_DIRECTORY + '/home-credit-default-risk/application_test.csv')
print('Testing data shape: ', app_test.shape().get())
app_test.head().get()

app_train['TARGET'].value_counts().get()

app_train['TARGET'].get().astype(int).plot.hist()


# Function to calculate missing values by column# Funct
def missing_values_table(dataset):
    # Total missing values
    mis_val = dataset.isnull().sum().get()

    mis_val_percent = 100 * mis_val / len(dataset.get())

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
    print("Your selected dataframe has " + str(dataset.shape().data[1]) + " columns.\n"
                                                                          "There are " + str(
        mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


missing_values = missing_values_table(app_train)
missing_values.head(20)

app_train.meta['dtypes'].value_counts()

app_train.select_dtypes('object').nunique().get()

# Create a label encoder object
le = LabelEncoder()
le_count = 0

for col in app_train.select_dtypes('object').get().columns:
    # we are not using nunique because it discard nan
    if app_train[col].nunique(dropna=False).get() <= 2:
        model = app_train[col].fit_sk_model(le)

        transformed_train = model.transform_col(app_train[col], col)
        app_train = app_train.drop(col)
        app_train = app_train.add_columns(col, transformed_train)

        transformed_test = model.transform_col(app_test[col], col)
        app_test = app_test.drop(col)
        app_test = app_test.add_columns(col, transformed_train)

        # Keep track of how many columns were label encoded
        le_count += 1
print('%d columns were label encoded.' % le_count)

app_train = app_train.onehot_encode()
app_test = app_test.onehot_encode()

print('Training Features shape: ', app_train.shape().get())
print('Testing Features shape: ', app_test.shape().get())

train_labels = app_train['TARGET']
for c in app_train.get().columns:
    if c not in app_test.get().columns:
        app_train = app_train.drop(c)

app_train = app_train.add_columns('TARGET', train_labels)

print('Training Features shape: ', app_train.shape().get())
print('Testing Features shape: ', app_test.shape().get())

(app_train['DAYS_BIRTH'] / 365).describe().get()

app_train['DAYS_EMPLOYED'].describe().get()

app_train['DAYS_EMPLOYED'].get().plot.hist(title='Days Employment Histogram')
plt.xlabel('Days Employment')

anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean().get()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean().get()))
print('There are %d anomalous days of employment' % anom.shape().get()[0])

days_employed_anom = app_train["DAYS_EMPLOYED"] == 365243
app_train = app_train.add_columns('DAYS_EMPLOYED_ANOM', days_employed_anom)
temp = app_train['DAYS_EMPLOYED'].replace({365243: np.nan})
app_train = app_train.drop('DAYS_EMPLOYED')
app_train = app_train.add_columns('DAYS_EMPLOYED', temp)

app_train["DAYS_EMPLOYED"].get().plot.hist(title='Days Employment Histogram');
plt.xlabel('Days Employment')

days_employed_anom = app_test["DAYS_EMPLOYED"] == 365243
app_test = app_test.add_columns('DAYS_EMPLOYED_ANOM', days_employed_anom)
temp = app_test['DAYS_EMPLOYED'].replace({365243: np.nan})
app_test = app_test.drop('DAYS_EMPLOYED')
app_test = app_test.add_columns('DAYS_EMPLOYED', temp)
print('There are %d anomalies in the test data out of %d entries'
      % (app_test['DAYS_EMPLOYED_ANOM'].sum().get(),
         app_test.shape().get()[0]))

correlations = app_train.corr().get()
top = correlations['TARGET'].sort_values()
# Display correlations
print('Most Positive Correlations:\n', top.tail(15))
print('\nMost Negative Correlations:\n', top.head(15))

abs_age = app_train['DAYS_BIRTH'].abs()
app_train = app_train.drop('DAYS_BIRTH')
app_train = app_train.add_columns('DAYS_BIRTH', abs_age)
app_train['DAYS_BIRTH'].corr(app_train['TARGET']).get()

# Set the style of plots
plt.style.use('fivethirtyeight')

# Plot the distribution of ages in years
plt.hist((app_train['DAYS_BIRTH'] / 365).get(), edgecolor='k', bins=25)
plt.title('Age of Client')
plt.xlabel('Age (years)')
plt.ylabel('Count')

plt.figure(figsize=(10, 8))
# KDE plot of loans that were repaid on time
sns.kdeplot((app_train[app_train['TARGET'] == 0]['DAYS_BIRTH'] / 365).get(), label='target == 0')
# KDE plot of loans which were not repaid on time
sns.kdeplot((app_train[app_train['TARGET'] == 1]['DAYS_BIRTH'] / 365).get(), label='target == 1')
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
age_data.head(10).get()

age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups.get()

plt.figure(figsize=(8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.get().index.astype(str), age_groups.get()['TARGET'] * 100)

# Plot labeling
plt.xticks(rotation=75);
plt.xlabel('Age Group (years)');
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')

ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr().get()
ext_data_corrs

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
    sns.kdeplot(source_data[app_train[column].notna()][column].get(), label='target == 0')
    # plot loans that were not repaid
    source_data = app_train[[column, 'TARGET']][app_train['TARGET'] == 1]
    sns.kdeplot(source_data[app_train[column].notna()][column].get(), label='target == 1')

    # Label the plots
    plt.title('Distribution of %s by Target Value' % column)
    plt.xlabel('%s' % column);
    plt.ylabel('Density');

plt.tight_layout(h_pad=2.5)

# Copy the data for plotting
plot_data = ext_data.drop('DAYS_BIRTH')

# Add in the age of the client in years
plot_data = plot_data.add_columns('YEARS_BIRTH', age_data['YEARS_BIRTH'])
# Drop na values and limit to first 100000 rows
plot_data = plot_data.head(100000).dropna()

# Create the pairgrid object
grid = sns.PairGrid(data=plot_data.get(), size=3, diag_sharey=False,
                    hue='TARGET',
                    vars=[x for x in list(plot_data.get().columns) if x != 'TARGET'])

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
imputer_model = poly_features.fit_sk_model(imputer)
poly_features = imputer_model.transform(poly_features)
poly_features_test = imputer_model.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree=3)

# Train the polynomial features
poly_transformer_model = poly_features.fit_sk_model(poly_transformer)

poly_features = poly_transformer_model.transform(poly_features)
poly_features_test = poly_transformer_model.transform(poly_features_test)

poly_transformer_model.get().get_feature_names(input_features=[
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'
])

# Call manually before meta data update
poly_features.get()
poly_features.set_columns(poly_transformer_model.get().get_feature_names(
    ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Add in the target
poly_features = poly_features.add_columns('TARGET', poly_target)

# Find the correlations with the target
poly_corrs = poly_features.corr().get()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))

# Call manually before meta data update
poly_features_test.get()
poly_features_test.set_columns(poly_transformer_model.get().get_feature_names(
    ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Merge polynomial features into training dataframe
poly_features = poly_features.add_columns('SK_ID_CURR',
                                          app_train['SK_ID_CURR'])
app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')

# Merge polnomial features into testing dataframe
poly_features_test = poly_features_test.add_columns('SK_ID_CURR',
                                                    app_test['SK_ID_CURR'])
app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

# Align the dataframes
for c in app_train_poly.get().columns:
    if c not in app_test_poly.get().columns:
        app_train_poly = app_train_poly.drop(c)

# # Print out the new shapes
print('Training data with polynomial features shape: ',
      app_train_poly.shape().get())
print('Testing data with polynomial features shape:  ',
      app_test_poly.shape().get())

app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain = app_train_domain.add_columns(
    'CREDIT_INCOME_PERCENT',
    app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL'])
app_train_domain = app_train_domain.add_columns(
    'ANNUITY_INCOME_PERCENT',
    app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL'])
app_train_domain = app_train_domain.add_columns(
    'CREDIT_TERM',
    app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT'])
app_train_domain = app_train_domain.add_columns(
    'DAYS_EMPLOYED_PERCENT',
    app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH'])

app_test_domain = app_test_domain.add_columns(
    'CREDIT_INCOME_PERCENT',
    app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL'])
app_test_domain = app_test_domain.add_columns(
    'ANNUITY_INCOME_PERCENT',
    app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL'])
app_test_domain = app_test_domain.add_columns(
    'CREDIT_TERM',
    app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT'])
app_test_domain = app_test_domain.add_columns(
    'DAYS_EMPLOYED_PERCENT',
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
        negative[app_train_domain[column].notna()][column].get(),
        label='target == 0')
    # plot loans that were not repaid
    positive = app_train_domain[[column, 'TARGET']][app_train['TARGET'] == 1]
    sns.kdeplot(
        positive[app_train_domain[column].notna()][column].get(),
        label='target == 1')

    # Label the plots
    plt.title('Distribution of %s by Target Value' % column)
    plt.xlabel('%s' % column)
    plt.ylabel('Density')

plt.tight_layout(h_pad=2.5)

from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in app_train.get().columns:
    train = app_train.drop(columns=['TARGET'])
else:
    train = app_train.copy()

# Feature names
features = list(train.get().columns)

# Copy of the testing data
test = app_test.copy()

# Median imputation of missing values
sk_imputer = Imputer(strategy='median')

# Scale each feature to 0-1
sk_scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
imputer = train.fit_sk_model(sk_imputer)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(test)

# Repeat with the scaler
scaler = train.fit_sk_model(sk_scaler)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape().get())
print('Testing data shape: ', test.shape().get())

from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
sk_log_reg = LogisticRegression(C=0.0001)

# Train on the training data
log_reg = train.fit_sk_model_with_labels(sk_log_reg, train_labels)

# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[1]

# Submission data
log_reg_pred.setname('TARGET')
submit = app_test['SK_ID_CURR'].concat(log_reg_pred)
submit.head().get()

from sklearn.ensemble import RandomForestClassifier

# Make the random forest classifier
sk_random_forest = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)

# Train on the training data
random_forest = train.fit_sk_model_with_labels(sk_random_forest, train_labels)

# Extract feature importances
feature_importances = random_forest.feature_importances(features)

# Make predictions on the test data
predictions = random_forest.predict_proba(test)[1]

# Score = 0.678
# Submission dataframe
predictions.setname('TARGET')
submit = app_test['SK_ID_CURR'].concat(predictions)
submit.head().get()

poly_features_names = list(app_train_poly.get().columns)

# Impute the polynomial features
sk_imputer = Imputer(strategy='median')
imputer = app_train_poly.fit_sk_model(sk_imputer)

poly_features = imputer.transform(app_train_poly)
poly_features_test = imputer.transform(app_test_poly)

# Scale the polynomial features
sk_scaler = MinMaxScaler(feature_range=(0, 1))
scaler = poly_features.fit_sk_model(sk_scaler)

poly_features = scaler.transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

sk_random_forest_poly = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
random_forest_poly = poly_features.fit_sk_model_with_labels(sk_random_forest_poly, train_labels)

# Make predictions on the test data
predictions = random_forest_poly.predict_proba(poly_features_test)[1]

# Score = 0.678
# Submission dataframe
predictions.setname('TARGET')
submit = app_test['SK_ID_CURR'].concat(predictions)
submit.head().get()

app_train_domain = app_train_domain.drop(columns='TARGET')

domain_features_names = list(app_train_domain.get().columns)

# Impute the domainnomial features
sk_imputer = Imputer(strategy='median')
imputer = app_train_domain.fit_sk_model(sk_imputer)

domain_features = imputer.transform(app_train_domain)
domain_features_test = imputer.transform(app_test_domain)

# Scale the domainnomial features
sk_scaler = MinMaxScaler(feature_range=(0, 1))
scaler = domain_features.fit_sk_model(sk_scaler)

domain_features = scaler.transform(domain_features)
domain_features_test = scaler.transform(domain_features_test)

# Train on the training data
sk_random_forest_domain = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
random_forest_domain = domain_features.fit_sk_model_with_labels(sk_random_forest_domain, train_labels)

# Extract feature importances
feature_importances_domain = random_forest_domain.feature_importances(domain_features_names)

# Make predictions on the test data
predictions = random_forest_domain.predict_proba(domain_features_test)[1]

# Score = 0.678
# Make a submission dataframe
predictions.setname('TARGET')
submit = app_test['SK_ID_CURR'].concat(predictions)
submit.head().get()


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
    df = df.add_columns('importance_normalized', df['importance'] / df['importance'].sum().get())

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.get().index[:15]))),
            df['importance_normalized'].get().head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.get().index[:15]))))
    ax.set_yticklabels(df['feature'].get().head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')
    plt.show()

    return df


# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)

feature_importances_domain_sorted = plot_feature_importances(feature_importances_domain)

# Save the Graph to Disk
# TODO: Maybe we need some versioning mechanism later on
ee.save_graph(GRAPH_LOCATION)
