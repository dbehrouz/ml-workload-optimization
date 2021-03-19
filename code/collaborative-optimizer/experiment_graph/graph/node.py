import copy
import hashlib
import uuid
from abc import abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiment_graph.graph.auxilary import DataFrame, DataSeries
# from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.benchmark_helper import BenchmarkMetrics
from experiment_graph.globals import COMBINE_OPERATION_IDENTIFIER

DEFAULT_RANDOM_STATE = 15071989
AS_KB = 1024.0


class Node(object):
    def __init__(self, node_id, execution_environment, underlying_data=None, size=None):
        self.id = node_id
        self.computed = False
        self.access_freq = 0
        self.execution_environment = execution_environment
        self.size = size
        self.underlying_data = underlying_data
        self.computed = False if underlying_data is None else True

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'execution_environment' in state:
            del state['execution_environment']
        return state

    def remove_content(self):
        del self.underlying_data
        self.underlying_data = None
        self.computed = False

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k is 'execution_environment':
                setattr(result, k, self.execution_environment)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def set_environment(self, environment):
        self.execution_environment = environment

    @abstractmethod
    def data(self, verbose):
        """
        verbose = 0 ==> no logging
        verbose = 1 ==> simple logging

        :param verbose:
        :return:
        """
        pass

    @abstractmethod
    def get_materialized_data(self):
        pass

    @abstractmethod
    def compute_size(self):
        pass

    @abstractmethod
    def clear_content(self):
        pass

    def update_freq(self):
        self.access_freq += 1

    def get_freq(self):
        return self.access_freq

    # TODO: when params are a dictionary with multiple keys the order may not be the same in str conversion
    @staticmethod
    def edge_hash(oper, params=''):
        return oper + '(' + str(params).replace(' ', '') + ')'

    @staticmethod
    def vertex_hash(prev, edge_hash):
        # TODO what are the chances that this cause a collision?
        # we should implement a collision strategy as well
        return hashlib.md5((prev + edge_hash).encode('utf-8')).hexdigest().upper()

    @staticmethod
    def generate_uuid():
        return uuid.uuid4().hex.upper()[0:8]

    @staticmethod
    def md5(val):
        return hashlib.md5(val.encode('utf-8')).hexdigest()

    @staticmethod
    def get_not_none(nextnode, exist):
        if exist is not None:
            return exist
        else:
            return nextnode

    # TODO: need to implement eager_mode when needed
    def generate_agg_node(self, oper, args=None, v_id=None, eager_mode=0):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        # nextid = self.generate_uuid()
        edge_hash = self.edge_hash(oper, args)
        nextid = self.vertex_hash(v_id, edge_hash)
        nextnode = Agg(nextid, self.execution_environment)
        exist = self.execution_environment.workload_dag.add_edge(v_id, nextid, nextnode,
                                                                 {'name': oper,
                                                                  'oper': 'p_' + oper,
                                                                  'execution_time': -1,
                                                                  'executed': False,
                                                                  'args': args,
                                                                  'hash': edge_hash},
                                                                 ntype=Agg.__name__)
        return self.get_not_none(nextnode, exist)

    def generate_groupby_node(self, oper, args=None, v_id=None):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        # nextid = self.generate_uuid()
        edge_hash = self.edge_hash(oper, args)
        nextid = self.vertex_hash(v_id, edge_hash)
        nextnode = GroupBy(nextid, self.execution_environment)
        exist = self.execution_environment.workload_dag.add_edge(v_id, nextid, nextnode,
                                                                 {'name': oper,
                                                                  'oper': 'p_' + oper,
                                                                  'execution_time': -1,
                                                                  'executed': False,
                                                                  'args': args,
                                                                  'hash': edge_hash},
                                                                 ntype=GroupBy.__name__)
        return self.get_not_none(nextnode, exist)

    def generate_sklearn_node(self, oper, args=None, v_id=None, should_warmstart=False):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        edge_arguments = dict()

        edge_arguments['should_warmstart'] = should_warmstart
        edge_arguments['warm_startable'] = hasattr(args['model'], 'warm_start')
        if edge_arguments['warm_startable']:
            if hasattr(args['model'], 'random_state'):
                edge_arguments['random_state'] = args['model'].random_state
                no_random_state_model = copy.deepcopy(args['model'])
                no_random_state_model.random_state = DEFAULT_RANDOM_STATE
                edge_arguments['no_random_state_model'] = str(no_random_state_model)
        edge_hash = self.edge_hash(oper, args)
        nextid = self.vertex_hash(v_id, edge_hash)
        nextnode = SK_Model(nextid, self.execution_environment)
        edge_arguments['name'] = type(args['model']).__name__
        edge_arguments['oper'] = 'p_' + oper
        edge_arguments['args'] = args
        edge_arguments['execution_time'] = -1
        edge_arguments['executed'] = False
        edge_arguments['hash'] = edge_hash
        exist = self.execution_environment.workload_dag.add_edge(v_id, nextid, nextnode,
                                                                 edge_arguments,
                                                                 ntype=SK_Model.__name__)
        return self.get_not_none(nextnode, exist)

    def generate_dataset_node(self, oper, args=None, v_id=None):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        # c_name = [] if c_name is None else c_name
        # c_hash = [] if c_hash is None else c_hash
        # nextid = self.generate_uuid()
        edge_hash = self.edge_hash(oper, args)
        nextid = self.vertex_hash(v_id, edge_hash)
        nextnode = Dataset(nextid, self.execution_environment)
        exist = self.execution_environment.workload_dag.add_edge(v_id, nextid, nextnode,
                                                                 {'name': oper,
                                                                  'oper': 'p_' + oper,
                                                                  'execution_time': -1,
                                                                  'executed': False,
                                                                  'args': args,
                                                                  'hash': edge_hash},
                                                                 ntype=Dataset.__name__)
        return self.get_not_none(nextnode, exist)

    def generate_feature_node(self, oper, args=None, v_id=None):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        # nextid = self.generate_uuid()
        edge_hash = self.edge_hash(oper, args)
        nextid = self.vertex_hash(v_id, edge_hash)
        nextnode = Feature(nextid, self.execution_environment)
        exist = self.execution_environment.workload_dag.add_edge(v_id, nextid, nextnode,
                                                                 {'name': oper,
                                                                  'execution_time': -1,
                                                                  'executed': False,
                                                                  'oper': 'p_' + oper,
                                                                  'args': args,
                                                                  'hash': edge_hash},
                                                                 ntype=type(nextnode).__name__)
        return self.get_not_none(nextnode, exist)

    def generate_evaluation_node(self, oper, args=None, v_id=None):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        edge_hash = self.edge_hash(oper, args)
        nextid = self.vertex_hash(v_id, edge_hash)
        nextnode = Evaluation(nextid, self.execution_environment)
        exist = self.execution_environment.workload_dag.add_edge(v_id, nextid, nextnode,
                                                                 {'name': oper,
                                                                  'execution_time': -1,
                                                                  'executed': False,
                                                                  'oper': 'p_' + oper,
                                                                  'args': args,
                                                                  'hash': edge_hash},
                                                                 ntype=type(nextnode).__name__)
        return self.get_not_none(nextnode, exist)

    def generate_super_node(self, nodes, args=None):
        args = {} if args is None else args
        involved_nodes = []
        for n in nodes:
            involved_nodes.append(n.id)
        # nextid = ''.join(involved_nodes)
        edge_hash = self.edge_hash(COMBINE_OPERATION_IDENTIFIER, args)
        nextid = self.vertex_hash(''.join(involved_nodes), edge_hash)
        if not self.execution_environment.workload_dag.has_node(nextid):
            nextnode = SuperNode(nextid, self.execution_environment, nodes)
            self.execution_environment.workload_dag.add_node(nextid,
                                                             **{'type': type(nextnode).__name__,
                                                                'root': False,
                                                                'data': nextnode,
                                                                'size': 0,
                                                                'involved_nodes': involved_nodes})
            for n in nodes:
                # this is to make sure each combined edge is a unique name
                # This is also used by the optimizer to find the other node when combine
                # edges are being examined
                self.execution_environment.workload_dag.graph.add_edge(n.id, nextid,
                                                                       # combine is a reserved word
                                                                       **{'name': COMBINE_OPERATION_IDENTIFIER,
                                                                          'oper': COMBINE_OPERATION_IDENTIFIER,
                                                                          'execution_time': -1,
                                                                          'executed': False,
                                                                          'args': args,
                                                                          'hash': edge_hash})
            return nextnode
        else:
            # TODO: add the update rule (even though it has no effect)
            return self.execution_environment.workload_dag.graph.nodes[nextid]['data']

    def run_operation(self, operation):
        """

        :type operation: Operation
        """
        return_type = operation.return_type
        if return_type == 'Dataset':
            self.generate_dataset_node(operation.name, operation.params)
        elif return_type == 'Feature':
            self.generate_feature_node(operation.name, operation.params)
        elif return_type == 'Agg':
            self.generate_agg_node(operation.name, operation.params)
        else:
            raise Exception('Invalid return type: {}'.format(return_type))

    def p_run_operation(self, operation):
        data = self.get_materialized_data()
        return operation.run(data)

    def store_dataframe(self, columns, df):
        self.execution_environment.data_storage.store_dataframe(columns, df)

    def store_feature(self, column, series):
        self.execution_environment.data_storage.store_dataseries(column, series)

    def find_column_index(self, c):
        return self.get_column().index(c)

    def get_c_hash(self, c):
        i = self.find_column_index(c)
        return self.get_column_hash()[i]

    def generate_hash(self, column_map, func_name):
        return {k: (self.md5(v + func_name)) for k, v in column_map.items()}

    def hash_and_return_dataframe(self, func_name, df, column_names=None, column_hashes=None):
        if column_names is None:
            column_names = self.get_column()
        if column_hashes is None:
            column_hashes = self.get_column_hash()
        new_c_hash = [(self.md5(v + func_name)) for v in column_hashes]
        # self.execution_environment.data_storage.store_dataset(new_c_hash, df[column_names])
        return DataFrame(column_names=column_names,
                         column_hashes=new_c_hash,
                         pandas_df=df[column_names])

    def hash_and_return_dataseries(self, func_name, series, c_name=None, c_hash=None):
        if c_name is None:
            c_name = self.get_column()
        if c_hash is None:
            c_hash = self.get_column_hash()
        new_c_hash = self.md5(c_hash + func_name)
        # self.execution_environment.data_storage.store_column(new_c_hash, series)
        return DataSeries(column_name=c_name,
                          column_hash=new_c_hash,
                          pandas_series=series)


class Agg(Node):

    def __init__(self, node_id, execution_environment, underlying_data=None, size=None):
        Node.__init__(self, node_id, execution_environment, underlying_data, size)

    def clear_content(self):
        del self.underlying_data
        self.underlying_data = None
        self.computed = False
        self.size = None

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.scheduler.schedule(
                self.execution_environment.experiment_graph,
                self.execution_environment.workload_dag,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.underlying_data

    def compute_size(self):
        if self.computed and self.size is None:
            start = datetime.now()
            from pympler import asizeof
            self.size = asizeof.asizeof(self.underlying_data) / AS_KB
            self.execution_environment.update_time(BenchmarkMetrics.NODE_SIZE_COMPUTATION,
                                                   (datetime.now() - start).total_seconds())
        return self.size

    def show(self):
        return self.id + " :" + self.get_materialized_data().__str__()


class Dataset(Node):
    """ Dataset class representing a dataset (set of Features)
    This class is analogous to pandas.core.frame.DataFrame

    TODO:
        * Integration with the graph library
        * Add support for every experiment_graph that Pandas DataFrame supports
        * Support for Python 3.x

    """

    def __init__(self, node_id, execution_environment, underlying_data=None, size=None):
        """

        :type underlying_data: DataFrame
        """
        Node.__init__(self, node_id, execution_environment, underlying_data, size)

    def clear_content(self):
        del self.underlying_data
        self.underlying_data = None
        self.computed = False
        self.size = None

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.scheduler.schedule(
                self.execution_environment.experiment_graph,
                self.execution_environment.workload_dag,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.underlying_data.pandas_df

    def get_column(self):
        return self.underlying_data.column_names

    def get_column_hash(self):
        return self.underlying_data.column_hashes

    def compute_size(self):
        if not self.computed:
            # This happens when compute_size is directly called by the user.
            self.data()
        if self.size is None:
            start = datetime.now()
            self.size = self.underlying_data.get_size()
            self.execution_environment.update_time(BenchmarkMetrics.NODE_SIZE_COMPUTATION,
                                                   (datetime.now() - start).total_seconds())
            return self.size
        else:
            return self.size

    def set_columns(self, columns):
        return self.generate_dataset_node('set_columns', {'columns': columns})

    def p_set_columns(self, columns):
        df = self.get_materialized_data().copy()
        # self.execution_environment.data_storage.store_dataset(self.c_hash, df)
        # df.columns = columns
        return DataFrame(column_names=columns,
                         column_hashes=self.get_column_hash(),
                         pandas_df=df)

    def rename(self, columns):
        return self.generate_dataset_node('rename', {'columns': columns})

    def p_rename(self, columns):
        df = self.get_materialized_data().rename(columns=columns)
        new_column_names = df.columns
        return DataFrame(column_names=new_column_names,
                         column_hashes=self.get_column_hash(),
                         pandas_df=df)

    def sample(self, n, random_state):
        return self.generate_dataset_node('sample', {'n': n, 'random_state': random_state})

    def p_sample(self, n, random_state):
        return self.hash_and_return_dataframe('sample{}{}'.format(n, random_state),
                                              self.get_materialized_data().sample(n=n, random_state=random_state))

    def project(self, columns):
        if type(columns) in [str, int]:
            return self.generate_feature_node('project', {'columns': columns})
        if type(columns) is list:
            return self.generate_dataset_node('project', {'columns': columns})

    def p_project(self, columns):
        if isinstance(columns, list):
            p_columns = []
            p_hashes = []
            for c in columns:
                p_columns.append(c)
                p_hashes.append(self.get_c_hash(c))
                # self.execution_environment.data_storage.store_dataset(p_hashes, self.get_materialized_data()[
                # p_columns])
            return DataFrame(column_names=p_columns,
                             column_hashes=p_hashes,
                             pandas_df=self.underlying_data.pandas_df[p_columns])
        else:
            # self.execution_environment.data_storage.store_column(self.get_c_hash(columns),
            #                                                      self.get_materialized_data()[columns])
            return DataSeries(column_name=columns,
                              column_hash=self.get_c_hash(columns),
                              pandas_series=self.underlying_data.pandas_df[columns])

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
        elif isinstance(index, Feature):
            supernode = self.generate_super_node([self, index], {'c_oper': 'filter_with'})
            return self.generate_dataset_node('filter_with', args={}, v_id=supernode.id)

        else:
            raise Exception('Unsupported operation. Only project (column index) is supported')

    def reset_index(self):
        return self.generate_dataset_node('reset_index')

    # TODO room for improvement
    # we re assigning a random hash to the new column, even though, if index column of two artifacts are the same
    # the content will be the same
    def p_reset_index(self):
        df = self.underlying_data.pandas_df.reset_index()
        new_column = df.columns[0]
        # new_column_data = df.iloc[0]
        new_c_hash = self.md5(self.generate_uuid())
        # self.execution_environment.data_storage.store_column(new_c_hash, new_column_data)
        # del df
        # del new_column_data

        return DataFrame(column_names=[new_column] + self.get_column(),
                         column_hashes=[new_c_hash] + self.get_column_hash(),
                         pandas_df=df)

    def copy(self):
        return self.generate_dataset_node('copy')

    def p_copy(self):
        return DataFrame(column_names=self.get_column(),
                         column_hashes=self.get_column_hash(),
                         pandas_df=self.underlying_data.pandas_df)

    def head(self, size=5):
        return self.generate_dataset_node('head', {'size': size})

    def p_head(self, size=5):
        return self.hash_and_return_dataframe('head{}'.format(size), self.get_materialized_data().head(size))

    def shape(self):
        return self.generate_agg_node('shape', {})

    def p_shape(self):
        return self.get_materialized_data().shape

    def isnull(self):
        return self.generate_dataset_node('isnull')

    def p_isnull(self):
        return self.hash_and_return_dataframe('isnull', self.get_materialized_data().isnull())

    def sum(self):
        return self.generate_agg_node('sum')

    def p_sum(self):
        return self.get_materialized_data().sum()

    def nunique(self, dropna=True):
        return self.generate_agg_node('nunique', {'dropna': dropna})

    def p_nunique(self, dropna):
        return self.get_materialized_data().nunique(dropna=dropna)

    def dtypes(self):
        return self.generate_agg_node('dtypes')

    def p_dtypes(self):
        return self.get_materialized_data().dtypes

    def describe(self):
        return self.generate_agg_node('describe')

    def p_describe(self):
        return self.get_materialized_data().describe()

    def abs(self):
        return self.generate_dataset_node('abs')

    def p_abs(self):
        return self.hash_and_return_dataframe('abs', self.get_materialized_data().abs())

    def mean(self):
        return self.generate_agg_node('mean')

    def p_mean(self):
        return self.get_materialized_data().mean()

    def min(self):
        return self.generate_agg_node('min')

    def p_min(self):
        return self.get_materialized_data().min()

    def max(self):
        return self.generate_agg_node('max')

    def p_max(self):
        return self.get_materialized_data().max()

    def count(self):
        return self.generate_agg_node('count')

    def p_count(self):
        return self.get_materialized_data().count()

    def std(self):
        return self.generate_agg_node('std')

    def p_std(self):
        return self.get_materialized_data().std()

    def quantile(self, values):
        return self.generate_agg_node('quantile', {'values': values})

    def p_quantile(self, values):
        return self.get_materialized_data().quantile(values=values)

    def notna(self):
        return self.generate_dataset_node('notna')

    def p_notna(self):
        return self.hash_and_return_dataframe('notna', self.get_materialized_data().notna())

    def select_dtypes(self, data_type):
        return self.generate_dataset_node('select_dtypes', {'data_type': data_type})

    # TODO: do a proper grouping of the methods
    # TODO: the dataframe shape and column names of some functions like select types and one hot encode
    # TODO: can only be inferred after the operation is executed. We should group these functions
    # (I think we are mentioning these in the paper as well)
    def p_select_dtypes(self, data_type):
        df = self.get_materialized_data().select_dtypes(data_type)
        c_names = []
        c_hashes = []
        # find the selected subset
        for c in df.columns:
            c_names.append(c)
            c_hashes.append(self.get_c_hash(c))
        # self.execution_environment.data_storage.store_dataset(c_hashes, df[c_names])
        return DataFrame(column_names=c_names,
                         column_hashes=c_hashes,
                         pandas_df=df[c_names])

    # If drop column results in one column the return type should be a Feature
    def drop(self, columns):
        return self.generate_dataset_node('drop', {'columns': columns})

    def p_drop(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        new_c = []
        new_hash = []
        for c in self.get_column():
            if c not in columns:
                new_c.append(c)
                new_hash.append(self.get_c_hash(c))
        # self.execution_environment.data_storage.store_dataset(new_hash, self.get_materialized_data()[new_c])
        return DataFrame(column_names=new_c,
                         column_hashes=new_hash,
                         pandas_df=self.get_materialized_data()[new_c])

    def dropna(self):
        return self.generate_dataset_node('dropna')

    def p_dropna(self):
        return self.hash_and_return_dataframe('dropna', self.get_materialized_data().dropna())

    def sort_values(self, col_name, ascending=False):
        return self.generate_dataset_node('sort_values', args={'col_name': col_name, 'ascending': ascending})

    def p_sort_values(self, col_name, ascending):
        return self.hash_and_return_dataframe('sort_values{}{}'.format(col_name, ascending),
                                              self.get_materialized_data().sort_values(col_name,
                                                                                       ascending=ascending).reset_index())

    def add_columns(self, col_names, features):
        if type(features) == list:
            raise Exception('Currently only one column at a time is allowed to be added')
            # supernode = self.generate_super_node([self] + features, {'col_names': col_names})
        else:
            supernode = self.generate_super_node([self, features], {'col_names': col_names, 'c_oper': 'add_columns'})

        return self.generate_dataset_node('add_columns', {'col_names': col_names}, v_id=supernode.id)

    def onehot_encode(self):
        return self.generate_dataset_node('onehot_encode', {})

    def p_onehot_encode(self):
        df = pd.get_dummies(self.get_materialized_data())
        new_column = []
        new_hash = []
        # create the md5 hash values for columns
        for c in df.columns:
            new_column.append(c)
            if c in self.get_column():
                new_hash.append(self.get_c_hash(c))
            else:
                # generate a unique prefix to make sure onehot encoding of different datasets with the same
                # column name does not result in the same column hash
                new_hash.append(self.md5(self.generate_uuid() + c))

        # self.execution_environment.data_storage.store_dataset(new_hash, df[new_column])
        return DataFrame(column_names=new_column,
                         column_hashes=new_hash,
                         pandas_df=df[new_column])

    def corr(self):
        return self.generate_agg_node('corr', {})

    def p_corr(self):
        return self.get_materialized_data().corr()

    # TODO: is it OK to assume as_index is always false?
    # we can recreate the original one anyway
    # For now we materialize the result as an aggregate node so everything will be stored inside the graph
    def groupby(self, col_names):
        return self.generate_groupby_node('groupby', {'col_names': col_names, 'as_index': False})

    # TODO create a new GroupBy underlying data type ( similar to DataFrame and DataSeries)
    def p_groupby(self, col_names, as_index):
        df = self.get_materialized_data().groupby(col_names, as_index=as_index)
        if isinstance(col_names, str):
            col_names = [col_names]
        key = self.md5("".join(col_names))
        new_key_columns = []
        new_key_hashes = []
        for c in col_names:
            new_key_columns.append(c)
            new_key_hashes.append(self.md5(self.get_c_hash(c) + 'groupkey' + key))

        new_group_columns = []
        new_group_hashes = []
        for c in self.get_column():
            # if it is , it's been added as part of the group key
            if c not in col_names:
                new_group_columns.append(c)
                new_group_hashes.append(self.md5(self.get_c_hash(c) + 'group' + key))

        return [df, new_key_columns, new_key_hashes, new_group_columns, new_group_hashes]

    # combined node
    def concat(self, nodes):
        if type(nodes) == list:
            supernode = self.generate_super_node([self] + nodes, {'c_oper': 'concat'})
        else:
            supernode = self.generate_super_node([self] + [nodes], {'c_oper': 'concat'})
        return self.generate_dataset_node('concat', v_id=supernode.id)

    # removes the columns that do not exist in the other dataframe
    def align(self, other):
        """
        Align is similar to pandas.align, however it only returns one dataset
        example Dataset.align:
            ds1.get_column() # ['1','2','3']
            ds2.get_column() # ['2','3','4']
            ds_aligned = ds1.align(ds2)
            ds_aligned.get_column() # ['2','3']
        example Pandas.align:
            df1.get_column() # ['1','2','3']
            df2.get_column() # ['2','3','4']
            df1_aligned,df2_aligned = ds1.align(ds2)
            df1_aligned.get_column() # ['2','3']
            df2_aligned.get_column() # ['2','3']

        If the user needs to align both datasets, he/she should call the function twice.
        Note: currently, we only support 'inner' join and column-oriented aligning
        :param other: other Dataset to align with
        :return: original Dataset with the columns that do not exist in the other dataset removed
        """
        supernode = self.generate_super_node([self, other], {'c_oper': 'align'})
        return self.generate_dataset_node('align', v_id=supernode.id)

    # dataframe merge operation operation of dataframes
    def merge(self, other, on, how='left'):
        supernode = self.generate_super_node([self, other], {'c_oper': 'merge'})
        return self.generate_dataset_node('merge', args={'on': on, 'how': how}, v_id=supernode.id)

    def fit_sk_model(self, model):
        return self.generate_sklearn_node('fit_sk_model', {'model': model})

    def fit_sk_model_with_labels(self, model, labels, custom_args=None, should_warmstart=False):
        supernode = self.generate_super_node([self, labels], {'c_oper': 'fit_sk_model_with_labels'})
        return self.generate_sklearn_node('fit_sk_model_with_labels', {'model': model, 'custom_args': custom_args},
                                          v_id=supernode.id, should_warmstart=should_warmstart)

    def p_fit_sk_model(self, model, warm_start=False):
        start = datetime.now()
        if warm_start:
            model.warm_start = True
        model.fit(self.get_materialized_data())
        self.execution_environment.update_time(BenchmarkMetrics.MODEL_TRAINING,
                                               (datetime.now() - start).total_seconds())
        return model

    def replace_columns(self, col_names, features):
        if type(features) == list:
            supernode = self.generate_super_node([self] + features,
                                                 {'col_names': col_names, 'c_oper': 'replace_columns'})
        else:
            supernode = self.generate_super_node([self, features],
                                                 {'col_names': col_names, 'c_oper': 'replace_columns'})

        return self.generate_dataset_node('replace_columns', {'col_names': col_names}, v_id=supernode.id)


class Evaluation(Node):

    def __init__(self, node_id, execution_environment, underlying_data=None, size=None):
        Node.__init__(self, node_id, execution_environment, underlying_data, size)

    def compute_size(self):
        if self.computed and self.size is None:
            start = datetime.now()
            from pympler import asizeof
            self.size = asizeof.asizeof(self.underlying_data) / AS_KB
            self.execution_environment.update_time(BenchmarkMetrics.NODE_SIZE_COMPUTATION,
                                                   (datetime.now() - start).total_seconds())
        return self.size

    def clear_content(self):
        del self.underlying_data
        self.underlying_data = None
        self.computed = False
        self.size = None

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.scheduler.schedule(
                self.execution_environment.experiment_graph,
                self.execution_environment.workload_dag,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.underlying_data


class Feature(Node):
    """ Feature class representing one (and only one) column of a data.
    This class is analogous to pandas.core.series.Series

    Todo:
        * Support for Python 3.x

    """

    def __init__(self, node_id, execution_environment, underlying_data=None, size=None):
        """

        :type underlying_data: DataSeries
        """
        Node.__init__(self, node_id, execution_environment, underlying_data, size)

    def clear_content(self):
        self.underlying_data = None
        self.computed = False
        self.size = None

    def dtype(self, verbose=0):
        if not self.computed:
            self.execution_environment.scheduler.schedule(
                self.execution_environment.experiment_graph,
                self.execution_environment.workload_dag,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data().dtype

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.scheduler.schedule(
                self.execution_environment.experiment_graph,
                self.execution_environment.workload_dag,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.underlying_data.pandas_series

    def get_column(self):
        return self.underlying_data.column_name

    def get_column_hash(self):
        return self.underlying_data.column_hash

    def compute_size(self):
        if self.computed and self.size is None:
            start = datetime.now()
            self.size = self.underlying_data.get_size()
            self.execution_environment.update_time(BenchmarkMetrics.NODE_SIZE_COMPUTATION,
                                                   (datetime.now() - start).total_seconds())
            return self.size
        else:
            return self.size

    def setname(self, name):
        return self.generate_feature_node('setname', {'name': name})

    def p_setname(self, name):
        return DataSeries(column_name=name,
                          column_hash=self.get_column_hash(),
                          pandas_series=self.underlying_data.pandas_series)

    def math(self, oper, other):
        # If other is a Feature Column
        if isinstance(other, Feature):
            supernode = self.generate_super_node([self, other], {'c_oper': oper})
            return self.generate_feature_node(oper, v_id=supernode.id)
        # If other is a numerical value
        else:
            return self.generate_feature_node(oper, {'other': other})

    # Overriding math operators
    def __mul__(self, other):
        return self.math('__mul__', other)

    def p___mul__(self, other):
        return self.hash_and_return_dataseries('__mul__{}'.format(other), self.get_materialized_data() * other)

    def __rmul__(self, other):
        return self.math('__rmul__', other)

    def p___rmul__(self, other):
        return self.hash_and_return_dataseries('__rmul__{}'.format(other), other * self.get_materialized_data())

    def __truediv__(self, other):
        return self.math('__truediv__', other)

    def p___truediv__(self, other):
        return self.hash_and_return_dataseries('__truediv__{}'.format(other), self.get_materialized_data() / other)

    def __rtruediv__(self, other):
        return self.math('__rtruediv__', other)

    def p___rtruediv__(self, other):
        return self.hash_and_return_dataseries('__rtruediv__{}'.format(other), other / self.get_materialized_data())

    def __itruediv__(self, other):
        return self.math('__itruediv__', other)

    def p___itruediv__(self, other):
        return self.hash_and_return_dataseries('__itruediv__{}'.format(other), other / self.get_materialized_data())

    def __add__(self, other):
        return self.math('__add__', other)

    def p___add__(self, other):
        return self.hash_and_return_dataseries('__add__{}'.format(other), self.get_materialized_data() + other)

    def __radd__(self, other):
        return self.math('__radd__', other)

    def p___radd__(self, other):
        return self.hash_and_return_dataseries('__radd__{}'.format(other), other + self.get_materialized_data())

    def __sub__(self, other):
        return self.math('__sub__', other)

    def p___sub__(self, other):
        return self.hash_and_return_dataseries('__sub_{}'.format(other), self.get_materialized_data() - other)

    def __rsub__(self, other):
        return self.math('__rsub__', other)

    def p___rsub__(self, other):
        return self.hash_and_return_dataseries('__rsub__{}'.format(other), other - self.get_materialized_data())

    def __lt__(self, other):
        return self.math('__lt__', other)

    def p___lt__(self, other):
        return self.hash_and_return_dataseries('__lt__{}'.format(other), self.get_materialized_data() < other)

    def __le__(self, other):
        return self.math('__le__', other)

    def p___le__(self, other):
        return self.hash_and_return_dataseries('__le__{}'.format(other), self.get_materialized_data() <= other)

    def __eq__(self, other):
        return self.math('__eq__', other)

    def p___eq__(self, other):
        return self.hash_and_return_dataseries('__eq__{}'.format(other), self.get_materialized_data() == other)

    def __ne__(self, other):
        return self.math('__ne__', other)

    def p___ne__(self, other):
        return self.hash_and_return_dataseries('__ne__{}'.format(other), self.get_materialized_data() != other)

    def __gt__(self, other):
        return self.math('__qt__', other)

    def p___gt__(self, other):
        return self.hash_and_return_dataseries('__gt__{}'.format(other), self.get_materialized_data() > other)

    def __ge__(self, other):
        return self.math('__qe__', other)

    def p___ge__(self, other):
        return self.hash_and_return_dataseries('__ge__{}'.format(other), self.get_materialized_data() >= other)

    # End of overridden methods

    def head(self, size=5):
        return self.generate_feature_node('head', {'size': size})

    def p_head(self, size=5):
        return self.hash_and_return_dataseries('head{}'.format(size), self.get_materialized_data().head(size))

    def fillna(self, value):
        return self.generate_feature_node('fillna', {'value': value})

    def p_fillna(self, value):
        return self.hash_and_return_dataseries('fill{}'.format(value), self.get_materialized_data().fillna(value))

    def astype(self, target_type):
        return self.generate_feature_node('astype', {'target_type': target_type})

    def p_astype(self, target_type):
        return self.hash_and_return_dataseries('astype{}'.format(target_type),
                                               self.get_materialized_data().astype(target_type))

    # combined node
    def concat(self, nodes):
        if type(nodes) == list:
            supernode = self.generate_super_node([self] + nodes, {'c_oper': 'concat'})
        else:
            supernode = self.generate_super_node([self] + [nodes], {'c_oper': 'concat'})
        return self.generate_dataset_node('concat', v_id=supernode.id)

    # TODO what happened here?
    def isnull(self):
        self.execution_environment.workload_dag.add_edge(self.id,
                                                         {'oper': self.edge_hash('isnull'),
                                                          'hash': self.edge_hash('isnull')},
                                                         ntype=type(self).__name__)

    def notna(self):
        return self.generate_feature_node('notna')

    def p_notna(self):
        return self.hash_and_return_dataseries('notna', self.get_materialized_data().notna())

    def sum(self):
        return self.generate_agg_node('sum')

    def p_sum(self):
        return self.get_materialized_data().sum()

    def nunique(self, dropna=True):
        return self.generate_agg_node('nunique', {'dropna': dropna})

    def p_nunique(self, dropna):
        return self.get_materialized_data().nunique(dropna=dropna)

    def describe(self):
        return self.generate_agg_node('describe')

    def p_describe(self):
        return self.get_materialized_data().describe()

    def mean(self):
        return self.generate_agg_node('mean')

    def p_mean(self):
        return self.get_materialized_data().mean()

    def median(self):
        return self.generate_agg_node('median')

    def p_median(self):
        return self.get_materialized_data().median()

    def min(self):
        return self.generate_agg_node('min')

    def p_min(self):
        return self.get_materialized_data().min()

    def max(self):
        return self.generate_agg_node('max')

    def p_max(self):
        return self.get_materialized_data().max()

    def count(self):
        return self.generate_agg_node('count')

    def p_count(self):
        return self.get_materialized_data().count()

    def std(self):
        return self.generate_agg_node('std')

    def p_std(self):
        return self.get_materialized_data().std()

    def quantile(self, values):
        return self.generate_agg_node('quantile', {'values': values})

    def p_quantile(self, values):
        return self.get_materialized_data().quantile(values=values)

    def value_counts(self):
        return self.generate_agg_node('value_counts')

    def p_value_counts(self):
        return self.get_materialized_data().value_counts()

    def abs(self):
        return self.generate_feature_node('abs')

    def p_abs(self):
        return self.hash_and_return_dataseries('abs', self.get_materialized_data().abs())

    def unique(self):
        return self.generate_feature_node('unique')

    def p_unique(self):
        return self.hash_and_return_dataseries('unique', self.get_materialized_data().unique())

    def dropna(self):
        return self.generate_feature_node('dropna')

    def p_dropna(self):
        return self.hash_and_return_dataseries('dropna', self.get_materialized_data().dropna())

    def binning(self, start_value, end_value, num):
        return self.generate_feature_node('binning',
                                          {'start_value': start_value, 'end_value': end_value, 'num': num})

    def p_binning(self, start_value, end_value, num):
        return self.hash_and_return_dataseries('binning{}{}{}'.format(start_value, end_value, num),
                                               pd.cut(self.get_materialized_data(),
                                                      bins=np.linspace(start_value, end_value, num=num)))

    def replace(self, to_replace):
        return self.generate_feature_node('replace', {'to_replace': to_replace})

    def p_replace(self, to_replace):
        return self.hash_and_return_dataseries('__replace__{}'.format(str(to_replace)),
                                               self.get_materialized_data().replace(to_replace, inplace=False))

    # def onehot_encode(self):
    #     self.execution_environment.graph.add_edge(self.id,
    #                                         {'oper': self.e_hash('onehot'), 'hash': self.e_hash('onehot')},
    #                                         ntype=Dataset.__name__)

    def corr(self, other):
        supernode = self.generate_super_node([self, other], {'c_oper': 'corr_with'})
        return self.generate_agg_node('corr_with', v_id=supernode.id)

    def fit_sk_model(self, model):
        return self.generate_sklearn_node('fit_sk_model', {'model': model})

    def p_fit_sk_model(self, model, warm_start=False):
        start = datetime.now()
        if warm_start:
            model.warm_start = True
        model.fit(self.get_materialized_data())
        self.execution_environment.update_time(BenchmarkMetrics.MODEL_TRAINING,
                                               (datetime.now() - start).total_seconds())
        return copy.deepcopy(model)


# TODO delete GroupBy node since they dont bring any benefit. seems they are lazily evaluated from the originating
# TODO dataset/Feature as a result we are just storing the data in the graph for no reason
class GroupBy(Node):

    def __init__(self, node_id, execution_environment, underlying_data=None, size=None):
        Node.__init__(self, node_id, execution_environment, underlying_data, size)

    def clear_content(self):
        del self.underlying_data
        self.underlying_data = None
        self.computed = False
        self.size = None

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.scheduler.schedule(
                self.execution_environment.experiment_graph,
                self.execution_environment.workload_dag,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.underlying_data

    def compute_size(self):
        if self.computed and self.size is None:
            raise Exception('Groupby objects size should be set using the set_size command first')

    def set_size(self, size):
        self.size = size

    def project(self, columns):
        return self.generate_groupby_node('project', {'columns': columns})

    def p_project(self, columns):
        data_object = self.get_materialized_data()
        df = data_object[0]
        key_columns = copy.copy(data_object[1])
        key_hashes = copy.copy(data_object[2])
        c_group_names = data_object[3]
        c_group_hashes = data_object[4]
        new_group_columns = []
        new_group_hashes = []
        if isinstance(columns, str):
            columns = [columns]
        for i in range(len(c_group_names)):
            if c_group_names[i] in columns:
                new_group_columns.append(c_group_names[i])
                new_group_hashes.append(c_group_hashes[i])
        return df[columns], key_columns, key_hashes, new_group_columns, new_group_hashes

    def __getitem__(self, index):
        return self.project(index)

    def count(self):
        return self.generate_dataset_node('count')

    def p_count(self):
        data_object = self.get_materialized_data()
        df = data_object[0].count()
        new_columns = copy.copy(data_object[1])
        new_hashes = copy.copy(data_object[2])

        c_group_names = data_object[3]
        c_group_hashes = data_object[4]

        # find the selected subset
        for i in range(len(c_group_names)):
            new_columns.append(c_group_names[i])
            new_hashes.append(self.md5(c_group_hashes[i] + 'count'))
        # self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return DataFrame(column_names=new_columns,
                         column_hashes=new_hashes,
                         pandas_df=df[new_columns])

    def agg(self, functions):
        return self.generate_dataset_node('agg', {'functions': functions})

    def p_agg(self, functions):
        data_object = self.get_materialized_data()
        # .reset_index() is essential as groupby (as_index=False) has no effect for agg function
        df = self.get_materialized_data()[0].agg(functions).reset_index()
        new_columns = copy.copy(data_object[1])
        new_hashes = copy.copy(data_object[2])

        c_group_names = data_object[3]
        c_group_hashes = data_object[4]

        # find the selected subset
        for i in range(len(c_group_names)):
            # Seems pandas automatically removes categorical variables for aggregations such as
            # mean, sum, and ...
            if c_group_names[i] in df.columns.levels[0]:
                for level2column in df.columns.levels[1]:
                    if level2column != '':
                        new_columns.append(c_group_names[i] + '_' + level2column)
                        new_hashes.append(self.md5(c_group_hashes[i] + level2column))

        df.columns = new_columns
        # self.execution_environment.data_storage.store_dataset(new_hashes, df)
        return DataFrame(column_names=new_columns,
                         column_hashes=new_hashes,
                         pandas_df=df)

    def mean(self):
        return self.generate_dataset_node('mean')

    def p_mean(self):
        data_object = self.get_materialized_data()
        df = data_object[0].mean()
        new_columns = copy.copy(data_object[1])
        new_hashes = copy.copy(data_object[2])

        c_group_names = data_object[3]
        c_group_hashes = data_object[4]

        # find the selected subset
        for i in range(len(c_group_names)):
            new_columns.append(c_group_names[i])
            new_hashes.append(self.md5(c_group_hashes[i] + 'mean'))

        # self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return DataFrame(column_names=new_columns,
                         column_hashes=new_hashes,
                         pandas_df=df[new_columns])


class SK_Model(Node):
    def __init__(self, node_id, execution_environment, underlying_data=None, size=None):
        Node.__init__(self, node_id, execution_environment, underlying_data, size)
        self.model_score = 0.0

    def clear_content(self):
        del self.underlying_data
        self.underlying_data = None
        self.computed = False
        self.size = None
        self.model_score = 0.0

    def set_model_score(self, model_score):
        self.model_score = model_score

    def get_model_score(self):
        return self.model_score

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.scheduler.schedule(
                self.execution_environment.experiment_graph,
                self.execution_environment.workload_dag,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.underlying_data

    def compute_size(self):
        if self.computed and self.size is None:
            start = datetime.now()

            if self.underlying_data.__class__.__name__ == 'RandomForestClassifier':
                # pympler returns the wrong size by a large margin for random forest
                # default random forest size is in the order 100s MBs where pympler returns
                # KBs
                import pickle
                import os
                with open(self.id, 'wb') as output:
                    pickle.dump(self.underlying_data, output, protocol=pickle.HIGHEST_PROTOCOL)
                self.size = os.stat(self.id).st_size / AS_KB
                os.remove(self.id)
            else:
                from pympler import asizeof
                self.size = asizeof.asizeof(self.underlying_data) / AS_KB
            self.execution_environment.update_time(BenchmarkMetrics.NODE_SIZE_COMPUTATION,
                                                   (datetime.now() - start).total_seconds())
        return self.size

    # The matching physical operator is in the supernode class
    def transform_col(self, node, col_name='NO_COLUMN_NAME'):
        supernode = self.generate_super_node([self, node], {'c_oper': 'transform_col'})
        return self.generate_feature_node('transform_col', args={'col_name': col_name}, v_id=supernode.id)

    def transform(self, node):
        supernode = self.generate_super_node([self, node], {'c_oper': 'transform'})
        return self.generate_dataset_node('transform', v_id=supernode.id)

    def feature_importances(self, domain_features_names=None):
        return self.generate_dataset_node('feature_importances',
                                          args={'domain_features_names': domain_features_names})

    def p_feature_importances(self, domain_features_names):
        if domain_features_names is None:
            df = pd.DataFrame({'feature': range(0, len(self.get_materialized_data().feature_importances_)),
                               'importance': self.get_materialized_data().feature_importances_})
        else:
            df = pd.DataFrame(
                {'feature': domain_features_names, 'importance': self.get_materialized_data().feature_importances_})

        new_columns = ['feature', 'importance']
        new_hashes = [self.md5(self.generate_uuid()) for c in new_columns]

        # self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return DataFrame(column_names=new_columns,
                         column_hashes=new_hashes,
                         pandas_df=df[new_columns])

    def predict_proba(self, test, custom_args=None):
        supernode = self.generate_super_node([self, test], {'c_oper': 'predict_proba'})
        return self.generate_dataset_node('predict_proba', args={'custom_args': custom_args}, v_id=supernode.id)

    def score(self, test, true_labels, score_type='accuracy', custom_args=None):
        supernode = self.generate_super_node([self, test, true_labels], {'c_oper': 'score'})
        return self.generate_evaluation_node('score', args={'score_type': score_type, 'custom_args': custom_args},
                                             v_id=supernode.id)


class SuperNode(Node):
    """SuperNode represents a (sorted) collection of other graph
    Its only purpose is to allow experiment_graph that require multiple graph to fit
    in our data model
    """

    def __init__(self, node_id, execution_environment, nodes):
        Node.__init__(self, node_id, execution_environment, underlying_data=None, size=0.0)
        self.nodes = nodes

    def clear_content(self):
        pass

    def compute_size(self):
        return 0.0

    def data(self, verbose):
        pass

    def get_materialized_data(self):
        pass

    def p_transform_col(self, col_name):
        model = self.nodes[0].get_materialized_data()
        feature_data = self.nodes[1].get_materialized_data()
        feature_hash = self.nodes[1].get_column_hash()
        new_hash = self.md5(feature_hash + str(model.get_params()))
        # self.execution_environment.data_storage.store_column(new_hash, pd.Series(model.transform(feature_data)))
        return DataSeries(column_name=col_name,
                          column_hash=new_hash,
                          pandas_series=pd.Series(model.transform(feature_data)))

    def p_transform(self):
        model = self.nodes[0].get_materialized_data()
        dataset_data = self.nodes[1].get_materialized_data()
        df = pd.DataFrame(model.transform(dataset_data))
        new_columns = df.columns
        new_hashes = [self.md5(self.generate_uuid()) for c in new_columns]
        # self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return DataFrame(column_names=new_columns,
                         column_hashes=new_hashes,
                         pandas_df=df[new_columns])

    def p_fit_sk_model_with_labels(self, model, custom_args, warm_start=False):
        start = datetime.now()
        if warm_start:
            model.warm_start = True
        if custom_args is None:
            model.fit(self.nodes[0].get_materialized_data(), self.nodes[1].get_materialized_data())
        else:
            model.fit(self.nodes[0].get_materialized_data(), self.nodes[1].get_materialized_data())
        # update the model training time in the graph
        self.execution_environment.update_time(BenchmarkMetrics.MODEL_TRAINING,
                                               (datetime.now() - start).total_seconds())
        return copy.deepcopy(model)

    def p_predict_proba(self, custom_args):
        if custom_args is None:
            df = self.nodes[0].get_materialized_data().predict_proba(self.nodes[1].get_materialized_data())
        else:
            df = self.nodes[0].get_materialized_data().predict_proba(self.nodes[1].get_materialized_data(),
                                                                     **custom_args)
        if hasattr(df, 'columns'):
            c_name = list(df.columns)
            c_hash = [self.md5(self.generate_uuid()) for c in c_name]
        else:
            c_name = [i for i in range(df.shape[1])]
            c_hash = [self.md5(self.generate_uuid()) for c in c_name]

        # self.execution_environment.data_storage.store_dataset(c_hash, d)
        return DataFrame(column_names=c_name,
                         column_hashes=c_hash,
                         pandas_df=pd.DataFrame(df, columns=c_name))

    def p_score(self, score_type, custom_args):
        model = self.nodes[0].get_materialized_data()
        test = self.nodes[1].get_materialized_data()
        true_labels = self.nodes[2].get_materialized_data()

        if score_type == 'accuracy':
            if custom_args is None:
                score = model.score(test, true_labels)
            else:
                score = model.score(test, true_labels, **custom_args)
        elif score_type == 'auc':
            if custom_args is None:
                predictions = model.predict_proba(test)[:, 1]
            else:
                predictions = model.predict_proba(test, **custom_args)[:, 1]
            score = roc_auc_score(true_labels, predictions)
        else:
            raise Exception('Score type \'{}\' is not supported'.format(score_type))

        # TODO to make things easier, we add score to the model
        self.nodes[0].set_model_score(score)
        return {score_type: score}

    def p_filter_with(self):
        return self.hash_and_return_dataframe('filter_with{}'.format(self.nodes[1].get_column_hash()),
                                              self.nodes[0].get_materialized_data()[
                                                  self.nodes[1].get_materialized_data()],
                                              self.nodes[0].get_column(),
                                              self.nodes[0].get_column_hash())

    def p_add_columns(self, col_names):
        # # already exists on the data storage
        # current_map = self.graph[0].map
        # # new data to be added
        # to_be_added = self.graph[1].map
        # if isinstance(col_names, list):
        #     return self.graph[0].data.assign(
        #         **dict(zip(col_names, [self.graph[1].data[a] for a in self.graph[1].data])))
        # else:
        #     return self.graph[0].data.assign(**dict(zip([col_names], [self.graph[1].data])))

        # Since both node 0 and 1 are already stored in the
        # TODO: This only works for adding one column at a time
        c_names = copy.copy(self.nodes[0].get_column())
        c_names.append(col_names)
        c_hash = copy.copy(self.nodes[0].get_column_hash())
        c_hash.append(copy.copy(self.nodes[1].get_column_hash()))
        data = pd.concat([self.nodes[0].get_materialized_data(), self.nodes[1].get_materialized_data()], axis=1)
        data.columns = c_names
        # self.execution_environment.data_storage.store_dataset(c_hash, data)
        return DataFrame(column_names=c_names,
                         column_hashes=c_hash,
                         pandas_df=data)

    def p_replace_columns(self, col_names):
        if isinstance(col_names, list):
            assert len(col_names) == len(self.nodes[1].get_column())
            c_names = copy.copy(self.nodes[0].get_column())
            c_hashes = copy.copy(self.nodes[0].get_column_hash())
            for i in range(len(col_names)):
                if col_names[i] not in c_names:
                    raise Exception('column {} does not exist in the dataset'.format(col_names[i]))
                index = self.nodes[0].find_column_index(col_names[i])
                c_hashes[index] = self.nodes[1].get_column_hash()[i]
        else:
            assert isinstance(self.nodes[1], Feature)
            c_names = copy.copy(self.nodes[0].get_column())
            c_hashes = copy.copy(self.nodes[0].get_column_hash())
            if col_names not in c_names:
                raise Exception('column {} does not exist in the dataset'.format(col_names))
            index = self.nodes[0].find_column_index(col_names)
            c_hashes[index] = self.nodes[1].get_column_hash()
        d1 = self.nodes[0].get_materialized_data()
        d2 = self.nodes[1].get_materialized_data()

        temp = d1.copy()
        temp[col_names] = d2
        # self.execution_environment.data_storage.store_dataset(c_hashes, d1[c_names])
        return DataFrame(column_names=c_names,
                         column_hashes=c_hashes,
                         pandas_df=d1[c_names])

    def p_corr_with(self):
        return self.nodes[0].get_materialized_data().corr(self.nodes[1].get_materialized_data())

    def p_concat(self):
        c_name = []
        c_hash = []
        for d in self.nodes:
            if isinstance(d, Feature):
                c_name = c_name + [d.get_column()]
                c_hash = c_hash + [d.get_column_hash()]
            elif isinstance(d, Dataset):
                c_name = c_name + d.get_column()
                c_hash = c_hash + d.get_column_hash()
            else:
                raise 'Cannot concatane object of type: {}'.format(type(d))
        data = pd.concat([self.nodes[0].get_materialized_data(), self.nodes[1].get_materialized_data()], axis=1)
        # self.execution_environment.data_storage.store_dataset(c_hash, data)
        return DataFrame(column_names=c_name,
                         column_hashes=c_hash,
                         pandas_df=data)

    # TODO: This can be done better
    # Now the assumption is, if it is a left join, then use the same hash of the left tables to avoid deduplication
    # If it is a right join use the hash of the right table
    # However, it is possible than for left or right join, the other table has it's old column values and we do not
    # need to store new columns in the data store
    # def p_merge(self, on, how):
    #     l_columns, l_hashes = self.graph[0].c_name, self.graph[0].c_hash
    #     r_columns, r_hashes = self.graph[1].c_name, self.graph[1].c_hash
    #     if how == 'left':
    #         print 'left join'
    #         new_columns = copy.deepcopy(l_columns)
    #         new_hashes = copy.deepcopy(l_hashes)
    #         for c in r_columns:
    #             if c != on:
    #                 new_columns.append(c)
    #                 new_hashes.append(self.md5(self.generate_uuid()))
    #     elif how == 'right':
    #         new_columns = copy.deepcopy(r_columns)
    #         new_hashes = copy.deepcopy(r_hashes)
    #         for c in l_columns:
    #             if c != on:
    #                 new_columns.append(c)
    #                 new_hashes.append(self.md5(self.generate_uuid()))
    #     else:
    #         new_columns = []
    #         new_hashes = []
    #         for c in l_columns:
    #             new_columns.append(c)
    #             new_hashes.append(self.md5(self.generate_uuid()))
    #         for c in r_columns:
    #             if c != on:
    #                 new_columns.append(c)
    #                 new_hashes.append(self.md5(self.generate_uuid()))
    #
    #     self.execution_environment.data_storage.store_dataframe(new_hashes,
    #                                                       self.graph[0].get_materialized_data().merge(self.graph[1].get_materialized_data(), on=on,
    #                                                                                  how=how)[new_columns])
    #     return new_columns, new_hashes

    # TODO: This is definitely not efficient as many of the columns maybe duplicated
    def p_merge(self, on, how):
        # perform the join
        df = self.nodes[0].get_materialized_data().merge(self.nodes[1].get_materialized_data(), on=on, how=how)
        # generate new hashes and store everything on disk
        new_columns = list(df.columns)
        new_hashes = [self.md5(self.generate_uuid()) for c in new_columns]
        # self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return DataFrame(column_names=new_columns,
                         column_hashes=new_hashes,
                         pandas_df=df[new_columns])

    def p_align(self):
        new_columns = []
        new_hashes = []
        current_columns = self.nodes[0].get_column()
        current_hashes = self.nodes[0].get_column_hash()
        for i in range(len(current_columns)):
            if current_columns[i] in self.nodes[1].get_column():
                new_columns.append(current_columns[i])
                new_hashes.append(current_hashes[i])
        # We only support for align with inner join on columns
        # df = self.nodes[0].get_materialized_data()[new_columns]
        # df = self.nodes[0].get_materialized_data().align(self.nodes[1].get_materialized_data(), join='inner', axis=1)
        return DataFrame(column_names=new_columns,
                         column_hashes=new_hashes,
                         pandas_df=self.nodes[0].get_materialized_data()[new_columns])

    # TODO: There may be a better way of hashing these so if the columns are copied and same operations are applied
    # we save storage space
    def p___div__(self):
        c_name = '__div__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__div__',
                                               self.nodes[0].get_materialized_data() / self.nodes[
                                                   1].get_materialized_data(),
                                               c_name, c_hash)

    def p___rdiv__(self):
        c_name = '__rdiv__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__rdiv__',
                                               self.nodes[1].get_materialized_data() / self.nodes[
                                                   0].get_materialized_data(),
                                               c_name, c_hash)

    def p___add__(self):
        c_name = '__add__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__add__',
                                               self.nodes[0].get_materialized_data() + self.nodes[
                                                   1].get_materialized_data(),
                                               c_name, c_hash)

    def p___radd__(self):
        c_name = '__radd__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__radd__',
                                               self.nodes[1].get_materialized_data() + self.nodes[
                                                   0].get_materialized_data(),
                                               c_name, c_hash)

    def p___sub__(self):
        c_name = '__sub__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__sub__',
                                               self.nodes[0].get_materialized_data() - self.nodes[
                                                   1].get_materialized_data(),
                                               c_name, c_hash)

    def p___rsub__(self):
        c_name = '__rsub__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__rub__',
                                               self.nodes[1].get_materialized_data() - self.nodes[
                                                   0].get_materialized_data(),
                                               c_name, c_hash)

    def p___lt__(self):
        c_name = '__lt__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__lt__',
                                               self.nodes[0].get_materialized_data() < self.nodes[
                                                   1].get_materialized_data(),
                                               c_name, c_hash)

    def p___le__(self):
        c_name = '__le__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__le__', self.nodes[0].get_materialized_data() <= self.nodes[
            1].get_materialized_data(), c_name, c_hash)

    def p___eq__(self):
        c_name = '__eq__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__eq__', self.nodes[0].get_materialized_data() == self.nodes[
            1].get_materialized_data(), c_name, c_hash)

    def p___ne__(self):
        c_name = '__ne__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__ne__', self.nodes[0].get_materialized_data() != self.nodes[
            1].get_materialized_data(), c_name, c_hash)

    def p___qt__(self):
        c_name = '__qt__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__qt__',
                                               self.nodes[0].get_materialized_data() > self.nodes[
                                                   1].get_materialized_data(),
                                               c_name, c_hash)

    def p___qe__(self):
        c_name = '__qe__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_return_dataseries('__qe__', self.nodes[0].get_materialized_data() >= self.nodes[
            1].get_materialized_data(), c_name, c_hash)
