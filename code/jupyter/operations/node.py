import pandas as pd
import numpy as np


class Node(object):
    def __init__(self, id, data):
        self.id = id
        self.data = data

    @staticmethod
    def load(loc, nrows=None):
        return Dataset(loc, pd.read_csv(loc, nrows=nrows))

    @staticmethod
    def merge(nodes):
        return SuperNode('super-node', nodes)


class Feature(Node):
    """ Feature class representing one (and only one) column of a data.
    This class is analogous to pandas.core.series.Series 
    
    Todo:
        * Integration with the graph library
        * Add support for every operations that Pandas Series supports
        * Support for Python 3.x
        
    """

    def __init__(self, id, data):
        Node.__init__(self, id, data)
        self.meta = {'name': data.name, 'dtype': data.dtype}

    def setname(self, name):
        self.data.name = name
        self.meta['name'] = name

    ### Overriding math operators
    def __mul__(self, other):
        return Feature(self.edge('multi', other), self.data * other)

    def __rmul__(self, other):
        return Feature(self.edge('multi', other), self.data * other)

    # TODO: When switching to python 3 this has to change to __floordiv__ and __truediv__
    def __div__(self, other):
        return Feature(self.edge('div', other), self.data / other)

    def __rdiv__(self, other):
        return Feature(self.edge('rdiv', other), other / self.data)

    def __add__(self, other):
        return Feature(self.edge('add', other), self.data + other)

    def __radd__(self, other):
        return Feature(self.edge('add', other), self.data + other)

    def __sub__(self, other):
        return Feature(self.edge('sub', other), self.data - other)

    def __rsub__(self, other):
        return Feature(self.edge('rsub', other), other - self.data)

    def __lt__(self, other):
        return Feature(self.edge('lt', other), self.data < other)

    def __le__(self, other):
        return Feature(self.edge('le', other), self.data <= other)

    def __eq__(self, other):
        return Feature(self.edge('eq', other), self.data == other)

    def __ne__(self, other):
        return Feature(self.edge('ne', other), self.data != other)

    def __qt__(self, other):
        return Feature(self.edge('qt', other), self.data > other)

    def __ge__(self, other):
        return Feature(self.edge('qe', other), self.data >= other)

    ### End of overriden methods

    def isnull(self):
        return Feature(self.edge('isnull'), self.data.isnull())

    def dropna(self):
        return Feature(self.edge('dropna'), self.data.dropna())

    def sum(self):
        return Agg(self.edge('sum'), self.data.sum())

    def nunique(self, dropna=True):
        return Agg(self.edge('nunique', dropna), self.data.nunique(dropna=dropna))

    def describe(self):
        return Agg(self.edge('describe'), self.data.describe())

    def abs(self):
        return Feature(self.edge('abs'), self.data.abs())

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

    def unique(self):
        return Feature(self.edge('unique'), pd.Series(self.data.unique(), name='u_' + self.meta['name']))

    def binning(self, start_value, end_value, num):
        return Feature(self.edge('binning', str(start_value) + ',' + str(end_value) + ',' + str(num)),
                       pd.cut(self.data, bins=np.linspace(start_value, end_value, num=num)))

    def replace(self, to_replace):
        return Feature(self.edge('replace'), self.data.replace(to_replace, inplace=False))

    def onehot_encode(self):
        return Dataset(self.edge('onehot'), pd.get_dummies(self.data))

    def corr(self, other):
        id = self.id + '->corr_with' + other.id + '->end'
        return Agg(id, self.data.corr(other.data))

    def fit_sk_model(self, model):
        model.fit(self.data)
        return SK_Model(self.edge('fit ' + str(model.get_params.im_class)), model)


class Dataset(Node):
    """ Dataset class representing a dataset (set of Features)
    This class is analogous to pandas.core.frame.DataFrame
    
    Todo:
        * Integration with the graph library
        * Add support for every operations that Pandas DataFrame supports
        * Support for Python 3.x
        
    """

    def __init__(self, id, data):
        Node.__init__(self, id, data)
        self.meta = {'columns': self.data.columns, 'dtypes': self.data.dtypes}

    def project(self, columns):
        id = self.edge('project', columns)
        if type(columns) is str:
            return Feature(id, self.data[columns])
        if type(columns) is list:
            return Dataset(id, self.data[columns])

    # overloading the indexing operator similar operation to project  
    def __getitem__(self, index):
        """ Overrides getitem method
        If the index argument is of type string or a list, we apply a projection operator (indexing columns)
        If the index argument is of type Feature, we apply a 'join' operator where we filter the data using values
        in the Feature. The data in the feature must be of the form (index, Boolean)
        """
        # project operator
        if type(index) in [str, list]:
            return self.project(index)
        # index operator using another Series of the form (index,Boolean)
        elif isinstance(index, Feature):
            id = self.id + '->filter-using' + index.id + '->end'
            return Dataset(id, self.data[index.data])
        else:
            raise Exception(
                'Unsupported operation. Only project (column index) and filter (row-index using boolean) are supported')

    def head(self, size=5):
        return Dataset(self.edge('head'), self.data.head(size))

    def shape(self):
        return Agg(self.edge('shape'), self.data.shape)

    def isnull(self):
        return Dataset(self.edge('isnull'), self.data.isnull())

    def sum(self):
        return Agg(self.edge('sum'), self.data.sum())

    def select_dtypes(self, data_type):
        return Dataset(self.edge('select_dtypes'), self.data.select_dtypes(data_type))

    def nunique(self, dropna=True):
        return Agg(self.edge('nunique', dropna), self.data.nunique(dropna=dropna))

    def drop_column(self, col_name):
        return Dataset(self.edge('drop', col_name), self.data.drop(columns=col_name))

    def onehot_encode(self):
        return Dataset(self.edge('onehot'), pd.get_dummies(self.data))

    def corr(self):
        return Agg(self.edge('corr'), self.data.corr())


class Agg(Node):
    def __init__(self, id, data):
        Node.__init__(self, id, data)

    def show(self):
        return self.id + " :" + self.data.__str__()


class SK_Model(Node):
    def __init__(self, id, data):
        Node.__init__(self, id, data)
        self.meta = data.get_params.im_class


class SuperNode(Node):
    """SuperNode represents a (sorted) collection of other nodes
    Its only purpose is to allow operations that require multiple nodes to fit 
    in our data model
    """

    def __init__(self, id, nodes):
        Node.__init__(self, id, None)
        self.nodes = nodes
        self.meta = {'count': len(nodes)}

    def concat(self):
        id = ''
        ds = []
        for d in self.nodes:
            id = id + '->concat' + d.id
            ds.append(d.data)
        id = id + '->end'
        return Dataset(id, pd.concat(ds, axis=1))

    # for this operation, the Supernode should contain two nodes,
    # the first node is the aggregated model and the second one
    # is the node that model will transform
    def transform_with_sk_model(self, col_name):
        return Feature(self.edge('transform ' + str(self.nodes[0].data.get_params.im_class)),
                       pd.Series(self.nodes[0].data.transform(self.nodes[1].data), name=col_name))
