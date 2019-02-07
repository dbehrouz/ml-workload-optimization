import pandas as pd 
import numpy as np

# Feature level operations
def project(data, column):
    return data[column]

def concat(columns):
    return pd.concat(columns, axis=1)

def f_isnull(column):
    return column.isnull()

def f_dropna(column):
    return column.dropna()

def f_sum(column):
    return pd.Series(column.sum(), name=column.name)

def f_nunique(column):
    return pd.Series(column.nunique(), name=column.name)

def f_multi(column, scalar):
    return column * scalar

def f_describe(column):
    return column.describe()

def f_mean(column):
    return column.mean()

def f_min(column):
    return column.min()

def f_max(column):
    return column.max()

def f_count(column):
    return column.count()

def f_std(column):
    return column.std()

def f_quantile(column,values):
    return column.quantile(values)

def f_filter_equal(column, value):
    return column[column==value]

def f_filter_not_equal(column, value):
    return column[column != value]

def f_replace_with_value(column, value):
    return column.replace(value)

def f_sort_values(column):
    return column.sort_values()

def f_abs(column):
    return abs(column)

def f_corr(column1, column2):
    return column1.corr(column2)

def f_binning(column, start_value, end_value, num):
    return pd.cut(column, bins = np.linspace(start_value, end_value, num= num))

def f_head(column, size):
    return column.head(size)
   
    
# Dataset level helper operations
# All of these operations are translated into column level operations
def d_sum(df):
    arr = []
    for c in df.columns:
        pc = project(df, c)
        pcsum = f_sum(pc)
        arr.append(pcsum)
    return concat(arr)

def d_nunique(df):
    arr = []
    for c in df.columns:
        pc = project(df, c)
        pcu = f_nunique(pc)
        arr.append(pcu)
    return concat(arr)

def d_isnull(df):
    arr = []
    for c in df.columns:
        pc = project(df, c)
        pcin = f_isnull(pc)
        arr.append(pcin)
    return concat(arr)

def d_groupby_mean(df, column_name):
    return df.groupby(column_name).mean()

def d_dropna(df):
    return df.dropna()

def d_head(df, size):
    return df.head(size)

def d_corr(df):
    return df.corr()

def d_filter_rows_equal(df, column_name, value):
     return df[df[column_name] == value]
    
def d_filter_rows_not_equal(df, column_name, value):
     return df[df[column_name] != value]

def one_hot_encode(data):
    return pd.get_dummies(data)

def load_csv(loc, nrows= None):
    return pd.read_csv(loc, nrows=nrows)

def add_column(data, data_object, col_name):
    data[col_name] = data_object
    return data

def drop_column(data, col_name):
    return data.drop(columns=col_name)

def shape(data):
    return data.shape

def head(data):
    return data.head()

def value_counts(data):
    return data.value_counts()

def isnull(data):
    return data.isnull()
    
# can I make this generic (i.e., use the transform function instead)
def dtypes(data):
    return data.dtypes

# can I make this generic (some sort of filter function)
def select_dtypes(data, data_type):
    return data.select_dtypes(data_type)


# ML operations
def fit_sklearn_model(model, data):
    model.fit(data)
    return model

def apply_sklearn_model(model, data):
    return model.transform(data)

# should be changed once we have a join/split function
def align(left, right):
    return left.align(right, join = 'inner', axis = 1)
    
