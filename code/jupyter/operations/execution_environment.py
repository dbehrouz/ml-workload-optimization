import uuid

import pandas as pd

from execution_graph import ExecutionGraph


class ExecutionEnvironment(object):
    graph = ExecutionGraph()

    @staticmethod
    def load(loc, nrows=None):
        nextnode = ExecutionEnvironment.Dataset(loc, pd.read_csv(loc, nrows=nrows))
        ExecutionEnvironment.graph.roots.append(loc)
        ExecutionEnvironment.graph.add_node(loc, **{'root': True, 'type': 'Dataset', 'data': nextnode, 'loc': loc})
        return nextnode

    class Node(object):
        def __init__(self, id, data):
            self.id = id
            self.data = data
            self.meta = {}

        @staticmethod
        def merge(nodes):
            return SuperNode('super-node', nodes)

        # TODO: when params are a dictionary with multiple keys the order may not be the same in str conversion
        def e_hash(self, oper, params=''):
            return oper + '(' + str(params).replace(' ', '') + ')'

        def v_uuid(self):
            return uuid.uuid4().hex.upper()[0:8]

        def get(self):
            # compute and return the result
            # graph.compute_result(self.id)
            if self.is_empty():
                ExecutionEnvironment.graph.compute_result(self.id)
                self.update_meta()
            return self.data

        def getNotNone(self, nextnode, exist):
            if exist is not None:
                return exist
            else:
                return nextnode

        def is_empty(self):
            return self.data is None or 0 == len(self.data)

        def generate_agg_node(self, oper, args={}):
            nextid = self.v_uuid()
            nextnode = ExecutionEnvironment.Agg(nextid, None)
            exist = ExecutionEnvironment.graph.add_edge(self.id, nextid, nextnode,
                                                        {'name': oper,
                                                         'oper': 'p_' + oper,
                                                         'args': args,
                                                         'hash': self.e_hash(oper, args)},
                                                        ntype=ExecutionEnvironment.Agg.__name__)
            return self.getNotNone(nextnode, exist)

        def generate_sklearn_node(self, oper, args={}, id=None):
            if id is None:
                id = self.id
            nextid = self.v_uuid()
            nextnode = ExecutionEnvironment.SK_Model(nextid, None)
            exist = ExecutionEnvironment.graph.add_edge(id, nextid, nextnode,
                                                        {'name': oper,
                                                         'oper': 'p_' + oper,
                                                         'args': args,
                                                         'hash': self.e_hash(oper, args)},
                                                        ntype=ExecutionEnvironment.Agg.__name__)
            return self.getNotNone(nextnode, exist)

        def generate_dataset_node(self, oper, args={}, id=None):
            if id is None:
                id = self.id
            nextid = self.v_uuid()
            nextnode = ExecutionEnvironment.Dataset(nextid, pd.DataFrame())
            exist = ExecutionEnvironment.graph.add_edge(id, nextid, nextnode,
                                                        {'name': oper,
                                                         'oper': 'p_' + oper,
                                                         'args': args,
                                                         'hash': self.e_hash(oper, args)},
                                                        ntype=ExecutionEnvironment.Dataset.__name__)
            return self.getNotNone(nextnode, exist)

        def generate_feature_node(self, oper, args={}, id=None):
            if id is None:
                id = self.id
            nextid = self.v_uuid()
            nextnode = ExecutionEnvironment.Feature(nextid, pd.Series())
            exist = ExecutionEnvironment.graph.add_edge(id, nextid, nextnode,
                                                        {'name': oper,
                                                         'oper': 'p_' + oper,
                                                         'args': args,
                                                         'hash': self.e_hash(oper, args)},
                                                        ntype=type(nextnode).__name__)
            return self.getNotNone(nextnode, exist)

        def generate_super_node(self, nodes, args={}):
            nextid = ''
            for n in nodes:
                nextid += n.id

            if not ExecutionEnvironment.graph.has_node(nextid):
                nextnode = ExecutionEnvironment.SuperNode(nextid, nodes)
                ExecutionEnvironment.graph.add_node(nextid,
                                                    **{'type': type(nextnode).__name__,
                                                       'root': False,
                                                       'data': nextnode})
                for n in nodes:
                    ExecutionEnvironment.graph.add_edge(n.id, nextid, nextnode,
                                                        {'name': 'merge',
                                                         'oper': 'merge',
                                                         'args': {},
                                                         'hash': self.e_hash('merge', args)},
                                                        ntype=type(nextnode).__name__)
                return nextnode
            else:
                # TODO: add the update rule (even though it has no effect)
                return ExecutionEnvironment.graph.graph.nodes[nextid]['data']

    class Feature(Node):
        """ Feature class representing one (and only one) column of a data.
        This class is analogous to pandas.core.series.Series 

        Todo:
            * Integration with the graph library
            * Add support for every operations that Pandas Series supports
            * Support for Python 3.x

        """

        def __init__(self, id, data):
            ExecutionEnvironment.Node.__init__(self, id, data)
            if len(data) > 0:
                self.update_meta()

        def update_meta(self):
            self.meta = {'name': self.data.name, 'dtype': self.data.dtype}

        def setname(self, name):
            self.data.name = name
            self.meta['name'] = name

        ### Overriding math operators
        def __mul__(self, other):
            nextid = self.v_uuid()
            nextnode = Feature(nextid, pd.Series())
            ExecutionEnvironment.graph.add_edge(self.id, id,
                                                {'oper': '__mul__',
                                                 'hash': self.edge('__mul__', other)},
                                                ntype=type(self).__name__)
            return nextnode

        def __rmul__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('multi', other), 'hash': self.edge('multi', other)},
                                                ntype=type(self).__name__)

        # TODO: When switching to python 3 this has to change to __floordiv__ and __truediv__
        def __div__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('div', other), 'hash': self.edge('div', other)},
                                                ntype=type(self).__name__)

        def __rdiv__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('rdiv', other), 'hash': self.edge('rdiv', other)},
                                                ntype=type(self).__name__)

        def __add__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('add', other), 'hash': self.edge('add', other)}
                                                , ntype=type(self).__name__)

        def __radd__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('add', other), 'hash': self.edge('add', other)},
                                                ntype=type(self).__name__)

        def __sub__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('sub', other), 'hash': self.edge('sub', other)},
                                                ntype=type(self).__name__)

        def __rsub__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('rsub', other), 'hash': self.edge('rsub', other)},
                                                ntype=type(self).__name__)

        def __lt__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('lt', other), 'hash': self.edge('lt', other)}
                                                , ntype=type(self).__name__)

        def __le__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('le', other), 'hash': self.edge('le', other)},
                                                ntype=type(self).__name__)

        def __eq__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('eq', other), 'hash': self.edge('eq', other)},
                                                ntype=type(self).__name__)

        def __ne__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('ne', other), 'hash': self.edge('ne', other)},
                                                ntype=type(self).__name__)

        def __gt__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('gt', other), 'hash': self.edge('gt', other)},
                                                ntype=type(self).__name__)

        def __ge__(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('ge', other), 'hash': self.edge('ge', other)},
                                                ntype=type(self).__name__)

        ### End of overriden methods

        def isnull(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('isnull'), 'hash': self.edge('isnull')},
                                                ntype=type(self).__name__)

        def dropna(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('dropna'), 'hash': self.edge('dropna')},
                                                ntype=type(self).__name__)

        def sum(self):
            return self.generate_agg_node('sum')

        def p_sum(self, dropna):
            return self.data.sum()

        def nunique(self, dropna=True):
            return self.generate_agg_node('nunique', {'dropna': dropna})

        def p_nunique(self, dropna):
            return self.data.nunique(dropna=dropna)

        def describe(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('describe'), 'hash': self.edge('desrcibe')},
                                                ntype=Agg.__name__)

        def abs(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('abs'), 'hash': self.edge('abs')},
                                                ntype=type(self).__name__)

        def mean(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('mean'), 'hash': self.edge('mean')},
                                                ntype=Agg.__name__)

        def min(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('min'), 'hash': self.edge('min')},
                                                ntype=Agg.__name__)

        def max(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('max'), 'hash': self.edge('max')},
                                                ntype=Agg.__name__)

        def count(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('count'), 'hash': self.edge('count')},
                                                ntype=Agg.__name__)

        def std(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('std'), 'hash': self.edge('std')},
                                                ntype=Agg.__name__)

        def quantile(self, values):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('quantile'), 'hash': self.edge('quantile')},
                                                ntype=Agg.__name__)

        def filter_equal(self, value):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('filter'), 'hash': self.edge('filter', value)},
                                                ntype=type(self).__name__)

        def filter_not_equal(self, value):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('filter_not_equal'),
                                                 'hash': self.edge('filter_not_equal',
                                                                   value)},
                                                ntype=type(self).__name__)

        def value_counts(self):
            return self.generate_agg_node('value_counts')

        def p_value_counts(self):
            return self.data.value_counts()

        def unique(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('unique'), 'hash': self.edge('unique')},
                                                ntype=type(self).__name__)

        def binning(self, start_value, end_value, num):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('binning'),
                                                 'hash': self.edge('binning',
                                                                   str(start_value) + ',' + str(end_value) + ',' + str(
                                                                       num))},
                                                ntype=type(self).__name__)

        def replace(self, to_replace):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('replace'),
                                                 'hash': self.edge('replace', to_replace)},
                                                ntype=type(self).__name__)

        def onehot_encode(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('onehot'), 'hash': self.edge('onehot')},
                                                ntype=Dataset.__name__)

        def corr(self, other):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('corr'), 'hash': self.edge('corr')},
                                                ntype=Agg.__name__)

        def fit_sk_model(self, model):
            return self.generate_sklearn_node('fit_sk_model', {'model': model})

        def p_fit_sk_model(self, model):
            model.fit(self.data)
            return model

    class Dataset(Node):
        """ Dataset class representing a dataset (set of Features)
        This class is analogous to pandas.core.frame.DataFrame

        Todo:
            * Integration with the graph library
            * Add support for every operations that Pandas DataFrame supports
            * Support for Python 3.x

        """

        def __init__(self, id, data):
            ExecutionEnvironment.Node.__init__(self, id, data)
            if len(data) > 0:
                self.update_meta()

        def update_meta(self):
            self.meta = {'columns': self.data.columns, 'dtypes': self.data.dtypes}

        def project(self, columns):
            if type(columns) is str:
                return self.generate_feature_node('project', {'columns': columns})
            if type(columns) is list:
                return self.generate_dataset_node('project', {'columns': columns})

        def p_project(self, columns):
            return self.data[columns]

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
            #             elif isinstance(index, Feature):
            #                 id = self.id + '->filter-using' +  index.id + '->end'
            #                 return Dataset(id, self.data[index.data])
            else:
                raise Exception('Unsupported operation. Only project (column index) is supported')

        def head(self, size=5):
            return self.generate_dataset_node('head', {'size': size})

        def p_head(self, size=5):
            return self.data.head(size)

        def shape(self):
            return self.generate_agg_node('shape', {})

        def p_shape(self):
            return self.data.shape

        def isnull(self):
            return self.generate_dataset_node('isnull')

        def p_isnull(self):
            return self.data.isnull()

        def sum(self):
            return self.generate_agg_node('sum', {})

        def p_sum(self):
            return self.data.sum()

        def select_dtypes(self, data_type):
            return self.generate_dataset_node('select_dtypes', {'data_type': data_type})

        def p_select_dtypes(self, data_type):
            return self.data.select_dtypes(data_type)

        def nunique(self, dropna=True):
            return self.generate_agg_node('nunique', {'dropna': dropna})

        def p_nunique(self, dropna=True):
            return self.data.nunique(dropna=dropna)

        # If drop column results in one column the return type should be a Feature
        def drop_column(self, col_name):
            return self.generate_dataset_node('drop_column', {'col_name': col_name})

        def p_drop_column(self, col_name):
            return self.data.drop(columns=col_name)

        def onehot_encode(self):
            return self.generate_dataset_node('onehot_encode', {})

        def p_onehot_encode(self):
            return pd.get_dummies(self.data)

        def corr(self):
            return self.generate_agg_node('corr', {})

        def p_corr(self):
            return self.data.corr()

        # merge node
        def concat(self, nodes):
            if type(nodes) == list:
                supernode = self.generate_super_node([self] + nodes)
            else:
                supernode = self.generate_super_node([self] + [nodes])
            return self.generate_dataset_node('concat', id=supernode.id)

    class Agg(Node):
        def __init__(self, id, data):
            ExecutionEnvironment.Node.__init__(self, id, data)

        def is_empty(self):
            return self.data is None

        def update_meta(self):
            self.meta = {'type': 'aggregation'}

        def show(self):
            return self.id + " :" + self.data.__str__()

    class SK_Model(Node):
        def __init__(self, id, data):
            ExecutionEnvironment.Node.__init__(self, id, data)

        def is_empty(self):
            return self.data is None

        def update_meta(self):
            self.meta = {'model_class': self.data.get_params.im_class}

        # The matching physical operator is in the supernode class
        def transform_col(self, node, col_name):
            supernode = self.generate_super_node([self, node])
            return self.generate_feature_node('transform_col', args={'col_name': col_name}, id=supernode.id)

    class SuperNode(Node):
        """SuperNode represents a (sorted) collection of other nodes
        Its only purpose is to allow operations that require multiple nodes to fit 
        in our data model
        """

        def __init__(self, id, nodes):
            ExecutionEnvironment.Node.__init__(self, id, None)
            self.nodes = nodes

        def update_meta(self):
            self.meta = {'count': len(self.nodes)}

        def p_transform_col(self, col_name):
            return pd.Series(self.nodes[0].data.transform(self.nodes[1].data), name=col_name)

        def p_concat(self):
            ds = []
            for d in self.nodes:
                ds.append(d.data)
            return pd.concat(ds, axis=1)

        def concat(self):
            id = ''
            ds = []
            for d in self.nodes:
                id = id + ',' + d.id
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('concat'), 'hash': self.edge('concat', id)},
                                                ntype=Dataset.__name__)
