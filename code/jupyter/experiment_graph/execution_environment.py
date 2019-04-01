import uuid

import numpy as np
import pandas as pd

from execution_graph import ExecutionGraph

# Reserved word for representing super nodes.
# Do not use combine as an operation name
COMBINE_OPERATION_IDENTIFIER = 'combine'
RANDOM_STATE = 15071989
AS_MB = 1024 * 1024


class ExecutionEnvironment(object):
    graph = ExecutionGraph()

    @staticmethod
    def plot_graph(plt):
        ExecutionEnvironment.graph.plot_graph(plt)

    @staticmethod
    def get_artifacts_size():
        return ExecutionEnvironment.graph.get_total_size()

    @staticmethod
    def load(loc, nrows=None):
        nextnode = ExecutionEnvironment.Dataset(loc, pd.read_csv(loc, nrows=nrows))
        size = sum(nextnode.data.memory_usage(index=True, deep=True)) / AS_MB
        ExecutionEnvironment.graph.roots.append(loc)
        ExecutionEnvironment.graph.add_node(loc, **{'root': True, 'type': 'Dataset', 'data': nextnode, 'loc': loc,
                                                    'size': size})
        return nextnode

    class Node(object):
        def __init__(self, node_id, data):
            self.id = node_id
            self.data = data
            self.meta = {}

        # TODO: when params are a dictionary with multiple keys the order may not be the same in str conversion
        @staticmethod
        def e_hash(oper, params=''):
            return oper + '(' + str(params).replace(' ', '') + ')'

        @staticmethod
        def generate_uuid():
            return uuid.uuid4().hex.upper()[0:8]

        def get(self, verbose=0):
            # compute and return the result
            # graph.compute_result(self.id)
            if self.is_empty():
                ExecutionEnvironment.graph.compute_result(self.id, verbose)
                self.reapply_meta()
            return self.data

        def update_meta(self):
            raise Exception('Node object has no meta data')

        def reapply_meta(self):
            raise Exception('Node class should not have been instantiated')

        def getNotNone(self, nextnode, exist):
            if exist is not None:
                return exist
            else:
                return nextnode

        def is_empty(self):
            return self.data is None or 0 == len(self.data)

        # TODO: need to implement eager_mode when needed
        def generate_agg_node(self, oper, args={}, v_id=None, eager_mode=0):
            if v_id is None:
                v_id = self.id
            nextid = self.generate_uuid()
            nextnode = ExecutionEnvironment.Agg(nextid, None)
            exist = ExecutionEnvironment.graph.add_edge(v_id, nextid, nextnode,
                                                        {'name': oper,
                                                         'oper': 'p_' + oper,
                                                         'args': args,
                                                         'hash': self.e_hash(oper, args)},
                                                        ntype=ExecutionEnvironment.Agg.__name__)
            return self.getNotNone(nextnode, exist)

        def generate_sklearn_node(self, oper, args={}, v_id=None):
            if v_id is None:
                v_id = self.id
            nextid = self.generate_uuid()
            nextnode = ExecutionEnvironment.SK_Model(nextid, None)
            exist = ExecutionEnvironment.graph.add_edge(v_id, nextid, nextnode,
                                                        {'name': oper,
                                                         'oper': 'p_' + oper,
                                                         'args': args,
                                                         'hash': self.e_hash(oper, args)},
                                                        ntype=ExecutionEnvironment.Agg.__name__)
            return self.getNotNone(nextnode, exist)

        def generate_dataset_node(self, oper, args={}, v_id=None, meta={}):
            if v_id is None:
                v_id = self.id
            nextid = self.generate_uuid()
            nextnode = ExecutionEnvironment.Dataset(nextid, pd.DataFrame(), meta)
            exist = ExecutionEnvironment.graph.add_edge(v_id, nextid, nextnode,
                                                        {'name': oper,
                                                         'oper': 'p_' + oper,
                                                         'args': args,
                                                         'hash': self.e_hash(oper, args)},
                                                        ntype=ExecutionEnvironment.Dataset.__name__)
            return self.getNotNone(nextnode, exist)

        def generate_feature_node(self, oper, args={}, v_id=None):
            if v_id is None:
                v_id = self.id
            nextid = self.generate_uuid()
            nextnode = ExecutionEnvironment.Feature(nextid, pd.Series())
            exist = ExecutionEnvironment.graph.add_edge(v_id, nextid, nextnode,
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
                    # this is to make sure each combined edge is a unique name
                    args['uuid'] = self.generate_uuid()
                    ExecutionEnvironment.graph.add_edge(n.id, nextid, nextnode,
                                                        # combine is a reserved word
                                                        {'name': COMBINE_OPERATION_IDENTIFIER,
                                                         'oper': COMBINE_OPERATION_IDENTIFIER,
                                                         'args': {},
                                                         'hash': self.e_hash(COMBINE_OPERATION_IDENTIFIER, args)},
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
            * Add support for every experiment_graph that Pandas Series supports
            * Support for Python 3.x

        """

        def __init__(self, id, data):
            ExecutionEnvironment.Node.__init__(self, id, data)
            if len(data) > 0:
                self.update_meta()

        def update_meta(self):
            self.meta = {'name': self.data.name, 'dtype': self.data.dtype}

        def reapply_meta(self):
            if not self.is_empty() and 'name' in self.meta.keys():
                self.data.name = self.meta['name']
            self.update_meta()

        def setname(self, name):
            self.meta['name'] = name
            self.reapply_meta()

        def math(self, oper, other):
            # If other is a Feature Column
            if isinstance(other, ExecutionEnvironment.Feature):
                supernode = self.generate_super_node([self, other])
                return self.generate_feature_node(oper, v_id=supernode.id)
            # If other is a numberical value
            else:
                return self.generate_feature_node(oper, {'other': other})

        # Overriding math operators
        def __mul__(self, other):
            return self.math('__mul__', other)

        def p___mul__(self, other):
            return self.data * other

        def __rmul__(self, other):
            return self.math('__rmul__', other)

        def p___rmul__(self, other):
            return other * self.data

        # TODO: When switching to python 3 this has to change to __floordiv__ and __truediv__
        def __div__(self, other):
            return self.math('__div__', other)

        def p___div__(self, other):
            return self.data / other

        def __rdiv__(self, other):
            return self.math('__rdiv__', other)

        def p___rdiv__(self, other):
            return other / self.data

        def __add__(self, other):
            return self.math('__add__', other)

        def p___add__(self, other):
            return self.data + other

        def __radd__(self, other):
            return self.math('__radd__', other)

        def p___radd__(self, other):
            return other + self.data

        def __sub__(self, other):
            return self.math('__sub__', other)

        def p___sub__(self, other):
            return self.data - other

        def __rsub__(self, other):
            return self.math('__rsub__', other)

        def p___rsub__(self, other):
            return other - self.data

        def __lt__(self, other):
            return self.math('__lt__', other)

        def p___lt__(self, other):
            return self.data < other

        def __le__(self, other):
            return self.math('__le__', other)

        def p___le__(self, other):
            return self.data <= other

        def __eq__(self, other):
            return self.math('__eq__', other)

        def p___eq__(self, other):
            return self.data == other

        def __ne__(self, other):
            return self.math('__ne__', other)

        def p___ne__(self, other):
            return self.data != other

        def __gt__(self, other):
            return self.math('__qt__', other)

        def p___gt__(self, other):
            return self.data > other

        def __ge__(self, other):
            return self.math('__qe__', other)

        def p___ge__(self, other):
            return self.data >= other

        # End of overridden methods

        def head(self, size=5):
            return self.generate_feature_node('head', {'size': size})

        def p_head(self, size=5):
            return self.data.head(size)

        # combined node
        def concat(self, nodes):
            if type(nodes) == list:
                supernode = self.generate_super_node([self] + nodes)
            else:
                supernode = self.generate_super_node([self] + [nodes])
            return self.generate_dataset_node('concat', v_id=supernode.id)

        def isnull(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('isnull'), 'hash': self.edge('isnull')},
                                                ntype=type(self).__name__)

        def notna(self):
            return self.generate_feature_node('notna')

        def p_notna(self):
            return self.data.notna()

        def sum(self):
            return self.generate_agg_node('sum')

        def p_sum(self):
            return self.data.sum()

        def nunique(self, dropna=True):
            return self.generate_agg_node('nunique', {'dropna': dropna})

        def p_nunique(self, dropna):
            return self.data.nunique(dropna=dropna)

        def describe(self):
            return self.generate_agg_node('describe')

        def p_describe(self):
            return self.data.describe()

        def mean(self):
            return self.generate_agg_node('mean')

        def p_mean(self):
            return self.data.mean()

        def min(self):
            return self.generate_agg_node('min')

        def p_min(self):
            return self.data.min()

        def max(self):
            return self.generate_agg_node('max')

        def p_max(self):
            return self.data.max()

        def count(self):
            return self.generate_agg_node('count')

        def p_count(self):
            return self.data.count()

        def std(self):
            return self.generate_agg_node('std')

        def p_std(self):
            return self.data.std()

        def quantile(self, values):
            return self.generate_agg_node('quantile', {'values': values})

        def p_quantile(self, values):
            return self.data.quantile(values=values)

        def value_counts(self):
            return self.generate_agg_node('value_counts')

        def p_value_counts(self):
            return self.data.value_counts()

        def abs(self):
            return self.generate_feature_node('abs')

        def p_abs(self):
            return self.data.abs()

        def unique(self):
            return self.generate_feature_node('unique')

        def p_unique(self):
            return self.data.unique()

        def dropna(self):
            return self.generate_feature_node('dropna')

        def p_dropna(self):
            return self.data.dropna()

        def binning(self, start_value, end_value, num):
            return self.generate_feature_node('binning',
                                              {'start_value': start_value, 'end_value': end_value, 'num': num})

        def p_binning(self, start_value, end_value, num):
            return pd.cut(self.data, bins=np.linspace(start_value, end_value, num=num))

        def replace(self, to_replace):
            return self.generate_feature_node('replace', {'to_replace': to_replace})

        def p_replace(self, to_replace):
            return self.data.replace(to_replace, inplace=False)

        def onehot_encode(self):
            ExecutionEnvironment.graph.add_edge(self.id,
                                                {'oper': self.edge('onehot'), 'hash': self.edge('onehot')},
                                                ntype=ExecutionEnvironment.Dataset.__name__)

        def corr(self, other):
            supernode = self.generate_super_node([self, other])
            return self.generate_agg_node('corr_with', v_id=supernode.id)

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
            * Add support for every experiment_graph that Pandas DataFrame supports
            * Support for Python 3.x

        """

        def __init__(self, id, data, meta={}):
            ExecutionEnvironment.Node.__init__(self, id, data)
            self.meta = meta
            if len(data) > 0:
                self.update_meta()

        def set_columns(self, columns):
            self.meta['columns'] = columns
            self.reapply_meta()

        def update_meta(self):
            self.meta = {'columns': self.data.columns, 'dtypes': self.data.dtypes}

        def reapply_meta(self):
            if 'columns' in self.meta.keys():
                self.data.columns = self.meta['columns']
            self.update_meta()

        def project(self, columns):
            if type(columns) in [str, int]:
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
            TODO:
             check how to implement the set_column operation, i.e. dataset['new_column'] = new_feature
            """
            # project operator
            if type(index) in [str, int, list]:
                return self.project(index)
            # index operator using another Series of the form (index,Boolean)
            elif isinstance(index, ExecutionEnvironment.Feature):
                supernode = self.generate_super_node([self, index])
                return self.generate_dataset_node('filter_with', args={}, v_id=supernode.id)

            else:
                raise Exception('Unsupported operation. Only project (column index) is supported')

        def copy(self):
            return self.generate_dataset_node('copy', meta=self.meta)

        def p_copy(self):
            return self.data.copy()

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
            return self.generate_agg_node('sum')

        def p_sum(self):
            return self.data.sum()

        def nunique(self, dropna=True):
            return self.generate_agg_node('nunique', {'dropna': dropna})

        def p_nunique(self, dropna):
            return self.data.nunique(dropna=dropna)

        def describe(self):
            return self.generate_agg_node('describe')

        def p_describe(self):
            return self.data.describe()

        def abs(self):
            return self.generate_dataset_node('abs')

        def p_abs(self):
            return self.data.abs()

        def mean(self):
            return self.generate_agg_node('mean')

        def p_mean(self):
            return self.data.mean()

        def min(self):
            return self.generate_agg_node('min')

        def p_min(self):
            return self.data.min()

        def max(self):
            return self.generate_agg_node('max')

        def p_max(self):
            return self.data.max()

        def count(self):
            return self.generate_agg_node('count')

        def p_count(self):
            return self.data.count()

        def std(self):
            return self.generate_agg_node('std')

        def p_std(self):
            return self.data.std()

        def quantile(self, values):
            return self.generate_agg_node('quantile', {'values': values})

        def p_quantile(self, values):
            return self.data.quantile(values=values)

        def notna(self):
            return self.generate_dataset_node('notna')

        def p_notna(self):
            return self.data.notna()

        def select_dtypes(self, data_type):
            return self.generate_dataset_node('select_dtypes', {'data_type': data_type})

        def p_select_dtypes(self, data_type):
            return self.data.select_dtypes(data_type)

        # If drop column results in one column the return type should be a Feature
        def drop(self, columns):
            return self.generate_dataset_node('drop', {'columns': columns})

        def p_drop(self, columns):
            return self.data.drop(columns=columns)

        def dropna(self):
            return self.generate_dataset_node('dropna')

        def p_dropna(self):
            return self.data.dropna()

        def sort_values(self, col_name, ascending=False):
            return self.generate_dataset_node('sort_values', args={'col_name': col_name, 'ascending': ascending})

        def p_sort_values(self, col_name, ascending):
            return self.data.sort_values(col_name, ascending=ascending).reset_index()

        def add_columns(self, col_names, features):
            if type(features) == list:
                supernode = self.generate_super_node([self] + features, {'col_names': col_names})
            else:
                supernode = self.generate_super_node([self, features], {'col_names': col_names})

            return self.generate_dataset_node('add_columns', {'col_names': col_names}, v_id=supernode.id)

        def onehot_encode(self):
            return self.generate_dataset_node('onehot_encode', {})

        def p_onehot_encode(self):
            return pd.get_dummies(self.data)

        def corr(self):
            return self.generate_agg_node('corr', {})

        def p_corr(self):
            return self.data.corr()

        # TODO: Do we need to create special grouped nodes?
        # For now Dataset node is good enough since aggregation experiment_graph that exist on group
        # also exist in the Dataset
        def groupby(self, col_names):
            return self.generate_dataset_node('groupby', {'col_names': col_names})

        def p_groupby(self, col_names):
            return self.data.groupby(col_names)

        # combined node
        def concat(self, nodes):
            if type(nodes) == list:
                supernode = self.generate_super_node([self] + nodes)
            else:
                supernode = self.generate_super_node([self] + [nodes])
            return self.generate_dataset_node('concat', v_id=supernode.id)

        # dataframe merge operation operation of dataframes
        def merge(self, other, on, how='left'):
            supernode = self.generate_super_node([self, other])
            return self.generate_dataset_node('merge', args={'on': on, 'how': how}, v_id=supernode.id)

        def fit_sk_model(self, model):
            return self.generate_sklearn_node('fit_sk_model', {'model': model})

        def fit_sk_model_with_labels(self, model, labels, custom_args=None):
            supernode = self.generate_super_node([self, labels])
            return self.generate_sklearn_node('fit_sk_model_with_labels', {'model': model, 'custom_args': custom_args},
                                              v_id=supernode.id)

        def p_fit_sk_model(self, model):
            model.fit(self.data)
            return model

        def replace_columns(self, col_names, features):
            if type(features) == list:
                supernode = self.generate_super_node([self] + features, {'col_names': col_names})
            else:
                supernode = self.generate_super_node([self, features], {'col_names': col_names})

            return self.generate_dataset_node('replace_columns', {'col_names': col_names}, v_id=supernode.id)

    class Agg(Node):
        def __init__(self, id, data):
            ExecutionEnvironment.Node.__init__(self, id, data)

        def is_empty(self):
            return self.data is None

        def update_meta(self):
            self.meta = {'type': 'aggregation'}

        def reapply_meta(self):
            self.update_meta()

        def show(self):
            return self.id + " :" + self.data.__str__()

    class SK_Model(Node):
        def __init__(self, id, data):
            ExecutionEnvironment.Node.__init__(self, id, data)

        def is_empty(self):
            return self.data is None

        def update_meta(self):
            self.meta = {'model_class': self.data.get_params.im_class}

        def reapply_meta(self):
            self.update_meta()

        # The matching physical operator is in the supernode class
        def transform_col(self, node, col_name):
            supernode = self.generate_super_node([self, node])
            return self.generate_feature_node('transform_col', args={'col_name': col_name}, v_id=supernode.id)

        def transform(self, node):
            supernode = self.generate_super_node([self, node])
            return self.generate_dataset_node('transform', v_id=supernode.id)

        def feature_importances(self, domain_features_names=None):
            return self.generate_dataset_node('feature_importances',
                                              args={'domain_features_names': domain_features_names})

        def p_feature_importances(self, domain_features_names):
            if domain_features_names is None:
                return pd.DataFrame({'feature': range(0, len(self.data.feature_importances_)),
                                     'importance': self.data.feature_importances_})
            else:
                return pd.DataFrame({'feature': domain_features_names, 'importance': self.data.feature_importances_})

        def predict_proba(self, test, custom_args=None):
            supernode = self.generate_super_node([self, test])
            return self.generate_dataset_node('predict_proba', args={'custom_args': custom_args}, v_id=supernode.id)

    class SuperNode(Node):
        """SuperNode represents a (sorted) collection of other nodes
        Its only purpose is to allow experiment_graph that require multiple nodes to fit
        in our data model
        """

        def __init__(self, id, nodes):
            ExecutionEnvironment.Node.__init__(self, id, None)
            self.nodes = nodes
            self.meta = {'count': len(self.nodes)}

        def update_meta(self):
            self.meta = {'count': len(self.nodes)}

        def reapply_meta(self):
            self.update_meta()

        def p_transform_col(self, col_name):
            return pd.Series(self.nodes[0].data.transform(self.nodes[1].data), name=col_name)

        def p_transform(self):
            df = self.nodes[0].data.transform(self.nodes[1].data)
            if hasattr(df, 'columns'):
                return pd.DataFrame(df, columns=df.columns)
            else:
                return pd.DataFrame(df)

        def p_fit_sk_model_with_labels(self, model, custom_args):
            if custom_args is None:
                model.fit(self.nodes[0].data, self.nodes[1].data)
            else:
                model.fit(self.nodes[0].data, self.nodes[1].data, **custom_args)
            return model

        def p_predict_proba(self, custom_args):
            if custom_args is None:
                df = self.nodes[0].data.predict_proba(self.nodes[1].data)
            else:
                df = self.nodes[0].data.predict_proba(self.nodes[1].data, **custom_args)

            if hasattr(df, 'columns'):
                return pd.DataFrame(df, columns=df.columns)
            else:
                return pd.DataFrame(df)

        def p_filter_with(self):
            return self.nodes[0].data[self.nodes[1].data]

        def p_add_columns(self, col_names):
            t = self.nodes[0].data
            t[col_names] = self.nodes[1].data
            return t

        def p_replace_columns(self, col_names):
            t = self.nodes[0].data
            t[col_names] = self.nodes[1].data
            return t

        def p_corr_with(self):
            return self.nodes[0].data.corr(self.nodes[1].data)

        def p_concat(self):
            ds = []
            for d in self.nodes:
                ds.append(d.data)
            return pd.concat(ds, axis=1)

        def p_merge(self, on, how):
            return self.nodes[0].data.merge(self.nodes[1].data, on=on, how=how)

        def p___div__(self):
            return self.nodes[0].data / self.nodes[1].data

        def p___rdiv__(self):
            return self.nodes[1].data / self.nodes[0].data

        def p___add__(self):
            return self.nodes[0].data + self.nodes[1].data

        def p___radd__(self):
            return self.nodes[1].data + self.nodes[0].data

        def p___sub__(self):
            return self.nodes[0].data - self.nodes[1].data

        def p___rsub__(self):
            return self.nodes[1].data - self.nodes[0].data

        def p___lt__(self):
            return self.nodes[0].data < self.nodes[1].data

        def p___le__(self):
            return self.nodes[0].data <= self.nodes[1].data

        def p___eq__(self):
            return self.nodes[0].data == self.nodes[1].data

        def p___ne__(self):
            return self.nodes[0].data != self.nodes[1].data

        def p___qt__(self):
            return self.nodes[0].data > self.nodes[1].data

        def p___qe__(self):
            return self.nodes[0].data >= self.nodes[1].data
