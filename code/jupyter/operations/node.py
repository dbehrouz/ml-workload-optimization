import pandas as pd 
import numpy as np

class Node:
    def __init__(self, id, data):
        self.id = id
        self.data = data
    
    def edge(self, oper, params=''):
        return self.id + '->' + oper + '(' + str(params).replace(' ', '') + ')'
    
    @staticmethod
    def load(loc, nrows = None): 
        return Dataset(loc, pd.read_csv(loc, nrows=nrows))
    

class Feature(Node):
    def __init__(self, id, data):
        Node.__init__(self, id, data)
        self.meta = {'name':data.name, 'dtype':data.dtype}
    
    def isnull(self):
        return Feature(self.edge('isnull'), self.data.isnull())
    
    def dropna(self):
        return Feature(self.edge('dropna'), self.data.dropna())
    
    def sum(self):
         return Feature(self.edge('sum'), pd.Series(self.data.sum(), name=self.data.name))
        
    def nunique(self):
        return Agg(self.edge('nunique'), self.data.nunique())
    
    def multi(self, scalar):
        id = self.edge('multi',scalar)
        return Feature(id, self.data * scalar)
    
    def describe(self):
        return Agg(self.edge('describe'), self.data.describe())
    
    def mean(self):
        return Agg(self.edge('mean'), self.data.mean())
    
    def min(self):
        return Agg(self.edge('min'), self.data.min())
    
    def max(self):
        return Agg(self.edge('max'), self.data.max())
    
    def count(self):
        return Agg(self.edge('count'), self.data.count())
    
    def std(self):
        return Agg(self.edge('std'), self.data.std())
    
    def quantile(self, values):
        return Agg(self.edge('quantile', values), self.data.quantile(values))
    
    def filter_equal(self, value):
        return Feature(self.edge('filter_equal', value), self.data[self.data == value])
    
    def filter_not_equal(self, value):
        return Feature(self.edge('filter_not_equal', value), self.data[self.data != value])
    
    def value_counts(self):
        return Agg(self.edge('value_counts'), self.data.value_counts())
    
    def fit_sk_model(self, model):
        model.fit(self.data)
        return SK_Model(self.edge('fit '+ str(model.get_params.im_class)), model)
    
    def transform_with_sk_model(self, model, col_name):
        return Feature(self.edge('transform '+ str(model.data.get_params.im_class)), 
                       pd.Series(model.data.transform(self.data), name=col_name))
        
                    
                   
class Dataset(Node):
    def __init__(self, id, data):
        Node.__init__(self, id, data)
        self.meta = self.data.dtypes
    
    def project(self, columns):
        id = self.edge('project', columns) 
        if type(columns) is str:
            return Feature(id, self.data[columns])
        if type(columns) is list:
            return Dataset(id, self.data[columns])
    
    def concat(self, other):
        id = self.id + '->concat' +  other.id + '->concat'
        return Dataset(id, pd.concat([self.data, other.data], axis = 1))
    
    def head(self, size=5):
        return Dataset(self.edge('head'), self.data.head(size))
    
    def shape(self):
        return Agg(self.edge('shape'),self.data.shape)
    
    def isnull(self):
        return Dataset(self.edge('isnull'), self.data.isnull())
    
    def sum(self):
        return Agg(self.edge('sum'), self.data.sum())
    
    def select_dtypes(self, data_type):
        return Dataset(self.edge('select_dtypes'), self.data.select_dtypes(data_type))
    
    def nunique(self):
        return Agg(self.edge('nunique'), self.data.nunique())
    
    def drop_column(self, col_name):
        return Dataset(self.edge('drop', col_name), self.data.drop(columns=col_name))
    
    
    @staticmethod
    def concat_many(datasets): 
        frames = []
        id = ''
        for d in datasets:
            frames.append(d.data)
            id = id + d.id+ self.edge('concat')
        return Dataset(id, pd.concat(frames), axis = 1)
    
class Agg(Node):
    def __init__(self, id, data):
        Node.__init__(self, id, data)
    
    def show(self):
        return self.id + " :" + self.data.__str__()
    
class SK_Model(Node):
    def __init__(self, id, data):
        Node.__init__(self, id, data)
        self.meta = data.get_params.im_class
    
    
    

   