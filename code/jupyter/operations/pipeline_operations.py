import pandas as pd 


def project(columns, data):
    return data[columns]

def set_column(data_object, col_name, data):
    data[col_name] = data_object
    return data

def shape(data):
    return data.shape

def head(data):
    return data.head()

def value_counts(data):
    return data.value_counts()
    
def aggregate(func, data):
    return data.agg(func)
   
def transform(func, data):
    return data.apply(func)
   
def load_csv(loc, nrows= None):
    return pd.read_csv(loc, nrows=nrows)

def concat(columns):
    return pd.concat(columns, axis=1)

# can I make this generic (i.e., use the transform function instead)
def dtypes(data):
    return data.dtypes

# can I make this generic (some sort of filter function)
def select_dtypes(data_type, data):
    return data.select_dtypes(data_type)

def fit_sklearn_model(model, data):
    model.fit(data)
    return model

def apply_sklearn_model(model, data):
    return model.transform(data)

def one_hot_encode(data):
    return pd.get_dummies(data)

# should be changed once we have a join/split function
def align(left, right):
    return left.align(right, join = 'inner', axis = 1)

def filter_rows(value, column, comparison, data):
    if comparison == '==':
        return data[data[column] == value]
    elif comparison == '!=':
        return data[data[column] != value]
    elif comparison == '>':
        return data[data[column] > value]
    elif comparison == '<':
        return data[data[column] < value]
    elif comparison == '>=':
        return data[data[column] >= value]
    elif comparison == '<=':
        return data[data[column] <= value]
    
