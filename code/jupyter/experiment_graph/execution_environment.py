import cPickle as pickle
import copy
import gc
import hashlib
import os
import uuid
from abc import abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd

from data_storage import DedupedStorageManager, NaiveStorageManager
from graph.execution_graph import ExecutionGraph, HistoryGraph, COMBINE_OPERATION_IDENTIFIER
# Reserved word for representing super graph.
# Do not use combine as an operation name
from optimizer import Optimizer

RANDOM_STATE = 15071989


class ExecutionEnvironment(object):
    def __init__(self, storage_type='dedup'):
        if storage_type == 'dedup':
            self.data_storage = DedupedStorageManager()
        elif storage_type == 'naive':
            self.data_storage = NaiveStorageManager()
        self.workload_graph = ExecutionGraph()
        self.history_graph = HistoryGraph()
        self.optimizer = Optimizer()
        self.time_manager = dict()

    def update_time(self, oper_type, seconds):
        if oper_type in self.time_manager:
            self.time_manager[oper_type] = self.time_manager[oper_type] + seconds
        else:
            self.time_manager[oper_type] = seconds

    def save_history(self, environment_folder, overwrite=False):
        if os.path.exists(environment_folder) and not overwrite:
            raise Exception('Directory already exists and overwrite is not allowed')
        if not os.path.exists(environment_folder):
            os.mkdir(environment_folder)

        with open(environment_folder + '/graph', 'wb') as output:
            self.history_graph.extend(self.workload_graph)
            pickle.dump(self.history_graph, output, pickle.HIGHEST_PROTOCOL)

        with open(environment_folder + '/storage', 'wb') as output:
            pickle.dump(self.data_storage, output, pickle.HIGHEST_PROTOCOL)

    def load_history(self, environment_folder):
        with open(environment_folder + '/graph', 'rb') as g_input:
            self.history_graph = pickle.load(g_input)
        with open(environment_folder + '/storage', 'rb') as d_input:
            self.data_storage = pickle.load(d_input)
            self.time_manager = dict()

    def plot_graph(self, plt, graph_type='workload'):
        if graph_type == 'workload':
            self.workload_graph.plot_graph(plt)
        else:
            self.history_graph.plot_graph(plt)

    def get_artifacts_size(self, graph_type='workload'):
        if graph_type == 'workload':
            return self.workload_graph.get_total_size()
        else:
            return self.history_graph.get_total_size()

    def load(self, loc, nrows=None):
        if self.workload_graph.has_node(loc):
            return self.workload_graph.get_node(loc)['data']
        else:
            initial_data = pd.read_csv(loc, nrows=nrows)
            c_name = []
            c_hash = []
            # create the md5 hash values for columns
            for i in range(len(initial_data.columns)):
                c = initial_data.columns[i]
                c_name.append(c)
                # to ensure the same hashing is used for the initial data loading and the rest of the artifacts
                # Adding the loc to make sure different datasets with the same column names do not mix
                c_hash.append(Node.md5(loc + c))

                self.data_storage.store_dataset(c_hash, initial_data[c_name])
            nextnode = Dataset(loc, self, c_name=c_name, c_hash=c_hash)
            size = self.data_storage.get_size(c_hash)
            self.workload_graph.roots.append(loc)
            self.workload_graph.add_node(loc, **{'root': True, 'type': 'Dataset', 'data': nextnode,
                                                 'loc': loc,
                                                 'size': size})
            return nextnode

    def load_from_pandas(self, df, identifier):
        if self.workload_graph.has_node(identifier):
            return self.workload_graph.get_node(identifier)['data']
        else:
            c_name = []
            c_hash = []
            # create the md5 hash values for columns
            for i in range(len(df.columns)):
                c = df.columns[i]
                c_name.append(c)
                # to ensure the same hashing is used for the initial data loading and the rest of the artifacts
                # Adding the loc to make sure different datasets with the same column names do not mix
                c_hash.append(Node.md5(identifier + c))
                self.data_storage.store_dataset(c_hash, df[c_name])
            nextnode = Dataset(identifier, self, c_name=c_name, c_hash=c_hash)
            size = self.data_storage.get_size(c_hash)
            self.workload_graph.roots.append(identifier)
            self.workload_graph.add_node(identifier,
                                         **{'root': True, 'type': 'Dataset', 'data': nextnode,
                                            'loc': identifier,
                                            'size': size})
            return nextnode


class Node(object):
    def __init__(self, node_id, execution_environment):
        self.id = node_id
        self.meta = {}
        self.computed = False
        self.access_freq = 0
        self.execution_environment = execution_environment

    @abstractmethod
    def data(self, verbose):
        pass

    @abstractmethod
    def get_materialized_data(self):
        pass

    def update_freq(self):
        self.access_freq += 1

    def get_freq(self):
        return self.access_freq

    # TODO: when params are a dictionary with multiple keys the order may not be the same in str conversion
    @staticmethod
    def e_hash(oper, params=''):
        return oper + '(' + str(params).replace(' ', '') + ')'

    @staticmethod
    def generate_uuid():
        return uuid.uuid4().hex.upper()[0:8]

    @staticmethod
    def md5(val):
        return hashlib.md5(val).hexdigest()

    def update_meta(self):
        raise Exception('Node object has no meta data')

    def reapply_meta(self):
        raise Exception('Node class should not have been instantiated')

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
        nextid = self.generate_uuid()
        nextnode = Agg(nextid, self.execution_environment)
        exist = self.execution_environment.workload_graph.add_edge(v_id, nextid, nextnode,
                                                                   {'name': oper,
                                                                    'oper': 'p_' + oper,
                                                                    'args': args,
                                                                    'hash': self.e_hash(oper, args)},
                                                                   ntype=Agg.__name__)
        return self.get_not_none(nextnode, exist)

    def generate_groupby_node(self, oper, args=None, v_id=None):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        nextid = self.generate_uuid()
        nextnode = GroupBy(nextid, self.execution_environment)
        exist = self.execution_environment.workload_graph.add_edge(v_id, nextid, nextnode,
                                                                   {'name': oper,
                                                                    'oper': 'p_' + oper,
                                                                    'args': args,
                                                                    'hash': self.e_hash(oper, args)},
                                                                   ntype=GroupBy.__name__)
        return self.get_not_none(nextnode, exist)

    def generate_sklearn_node(self, oper, args=None, v_id=None):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        nextid = self.generate_uuid()
        nextnode = SK_Model(nextid, self.execution_environment)
        exist = self.execution_environment.workload_graph.add_edge(v_id, nextid, nextnode,
                                                                   {'name': type(args['model']).__name__,
                                                                    'oper': 'p_' + oper,
                                                                    'args': args,
                                                                    'execution_time': -1,
                                                                    'hash': self.e_hash(oper, args)},
                                                                   ntype=SK_Model.__name__)
        return self.get_not_none(nextnode, exist)

    def generate_dataset_node(self, oper, args=None, v_id=None, c_name=None, c_hash=None):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        c_name = [] if c_name is None else c_name
        c_hash = [] if c_hash is None else c_hash
        nextid = self.generate_uuid()
        nextnode = Dataset(nextid, self.execution_environment, c_name, c_hash)
        exist = self.execution_environment.workload_graph.add_edge(v_id, nextid, nextnode,
                                                                   {'name': oper,
                                                                    'oper': 'p_' + oper,
                                                                    'execution_time': -1,
                                                                    'args': args,
                                                                    'hash': self.e_hash(oper, args)},
                                                                   ntype=Dataset.__name__)
        return self.get_not_none(nextnode, exist)

    def generate_feature_node(self, oper, args=None, v_id=None, c_name='', c_hash=''):
        v_id = self.id if v_id is None else v_id
        args = {} if args is None else args
        nextid = self.generate_uuid()
        nextnode = Feature(nextid, self.execution_environment, c_name, c_hash)
        exist = self.execution_environment.workload_graph.add_edge(v_id, nextid, nextnode,
                                                                   {'name': oper,
                                                                    'execution_time': -1,
                                                                    'oper': 'p_' + oper,
                                                                    'args': args,
                                                                    'hash': self.e_hash(oper, args)},
                                                                   ntype=type(nextnode).__name__)
        return self.get_not_none(nextnode, exist)

    def generate_super_node(self, nodes, args=None):
        args = {} if args is None else args
        involved_nodes = []
        for n in nodes:
            involved_nodes.append(n.id)
        nextnode_id = ''.join(involved_nodes)
        if not self.execution_environment.workload_graph.has_node(nextnode_id):
            nextnode = SuperNode(nextnode_id, self.execution_environment, nodes)
            self.execution_environment.workload_graph.add_node(nextnode_id,
                                                               **{'type': type(nextnode).__name__,
                                                                  'root': False,
                                                                  'data': nextnode,
                                                                  'involved_nodes': involved_nodes})
            for n in nodes:
                # this is to make sure each combined edge is a unique name
                # This is also used by the optimizer to find the other node when combine
                # edges are being examined
                self.execution_environment.workload_graph.add_edge(n.id, nextnode_id, nextnode,
                                                                   # combine is a reserved word
                                                                   {'name': COMBINE_OPERATION_IDENTIFIER,
                                                                    'oper': COMBINE_OPERATION_IDENTIFIER,
                                                                    'execution_time': 0,
                                                                    'args': args,
                                                                    'hash': self.e_hash(COMBINE_OPERATION_IDENTIFIER,
                                                                                        args)},
                                                                   ntype=type(nextnode).__name__)
            return nextnode
        else:
            # TODO: add the update rule (even though it has no effect)
            return self.execution_environment.workload_graph.graph.nodes[nextnode_id]['data']

    def store_dataframe(self, columns, df):
        self.execution_environment.data_storage.store_dataset(columns, df)

    def store_feature(self, column, series):
        self.execution_environment.data_storage.store_column(column, series)

    def find_column_index(self, c):
        return self.c_name.index(c)

    def get_c_hash(self, c):
        i = self.find_column_index(c)
        return self.c_hash[i]

    def generate_hash(self, column_map, func_name):
        return {k: (self.md5(v + func_name)) for k, v in column_map.items()}

    def hash_and_store_df(self, func_name, df, c_name=None, c_hash=None):
        if c_name is None:
            c_name = self.c_name
        if c_hash is None:
            c_hash = self.c_hash
        new_c_hash = [(self.md5(v + func_name)) for v in c_hash]
        self.execution_environment.data_storage.store_dataset(new_c_hash, df[c_name])
        return c_name, new_c_hash

    def hash_and_store_series(self, func_name, series, c_name=None, c_hash=None):
        if c_name is None:
            c_name = self.c_name
        if c_hash is None:
            c_hash = self.c_hash
        new_c_hash = self.md5(c_hash + func_name)
        self.execution_environment.data_storage.store_column(new_c_hash, series)
        return c_name, new_c_hash


class SuperNode(Node):
    """SuperNode represents a (sorted) collection of other graph
    Its only purpose is to allow experiment_graph that require multiple graph to fit
    in our data model
    """

    def data(self, verbose):
        pass

    def get_materialized_data(self):
        pass

    def __init__(self, node_id, execution_environment, nodes):
        Node.__init__(self, node_id, execution_environment)
        self.nodes = nodes

    def p_transform_col(self, col_name):
        model = self.nodes[0].get_materialized_data()
        feature_data = self.nodes[1].get_materialized_data()
        feature_hash = self.nodes[1].c_hash
        new_hash = self.md5(feature_hash + str(model.get_params()))
        self.execution_environment.data_storage.store_column(new_hash, pd.Series(model.transform(feature_data)))
        return col_name, new_hash

    def p_transform(self):
        model = self.nodes[0].get_materialized_data()
        dataset_data = self.nodes[1].get_materialized_data()
        df = pd.DataFrame(model.transform(dataset_data))
        new_columns = df.columns
        new_hashes = [self.md5(self.generate_uuid()) for c in new_columns]
        self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return new_columns, new_hashes

    def p_fit_sk_model_with_labels(self, model, custom_args):
        start = datetime.now()
        if custom_args is None:
            model.fit(self.nodes[0].get_materialized_data(), self.nodes[1].get_materialized_data(), )
        else:
            model.fit(self.nodes[0].get_materialized_data(), self.nodes[1].get_materialized_data(), )
        # update the model training time in the graph
        self.execution_environment.update_time('model-training', (datetime.now() - start).total_seconds())
        return model

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
        d = pd.DataFrame(df, columns=c_name)
        self.execution_environment.data_storage.store_dataset(c_hash, d)
        return c_name, c_hash

    def p_filter_with(self):
        return self.hash_and_store_df('filter_with{}'.format(self.nodes[1].c_hash),
                                      self.nodes[0].get_materialized_data()[self.nodes[1].get_materialized_data()],
                                      self.nodes[0].c_name,
                                      self.nodes[0].c_hash)

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
        c_names = copy.deepcopy(self.nodes[0].c_name)
        c_names.append(col_names)
        c_hash = copy.deepcopy(self.nodes[0].c_hash)
        c_hash.append(copy.deepcopy(self.nodes[1].c_hash))
        data = pd.concat([self.nodes[0].get_materialized_data(), self.nodes[1].get_materialized_data()], axis=1)
        self.execution_environment.data_storage.store_dataset(c_hash, data)
        return c_names, c_hash

    def p_replace_columns(self, col_names):
        if isinstance(col_names, list):
            assert len(col_names) == len(self.nodes[1].c_name)
            c_names = copy.deepcopy(self.nodes[0].c_name)
            c_hashes = copy.deepcopy(self.nodes[0].c_hash)
            for i in range(len(col_names)):
                if col_names[i] not in c_names:
                    raise Exception('column {} does not exist in the dataset'.format(col_names[i]))
                index = self.nodes[0].find_column_index(col_names[i])
                c_hashes[index] = self.nodes[1].c_hash[i]
        else:
            assert isinstance(self.nodes[1], Feature)
            c_names = copy.deepcopy(self.nodes[0].c_name)
            c_hashes = copy.deepcopy(self.nodes[0].c_hash)
            if col_names not in c_names:
                raise Exception('column {} does not exist in the dataset'.format(col_names))
            index = self.nodes[0].find_column_index(col_names)
            c_hashes[index] = self.nodes[1].c_hash
        d1 = self.nodes[0].get_materialized_data()
        d2 = self.nodes[1].get_materialized_data()
        d1[col_names] = d2
        self.execution_environment.data_storage.store_dataset(c_hashes, d1[c_names])
        return c_names, c_hashes

    def p_corr_with(self):
        return self.nodes[0].get_materialized_data().corr(self.nodes[1].get_materialized_data())

    def p_concat(self):
        c_name = []
        c_hash = []
        for d in self.nodes:
            if isinstance(d, Feature):
                c_name = c_name + [d.c_name]
                c_hash = c_hash + [d.c_hash]
            elif isinstance(d, Dataset):
                c_name = c_name + d.c_name
                c_hash = c_hash + d.c_hash
            else:
                raise 'Cannot concatane object of type: {}'.format(type(d))
        data = pd.concat([self.nodes[0].get_materialized_data(), self.nodes[1].get_materialized_data()], axis=1)
        self.execution_environment.data_storage.store_dataset(c_hash, data)
        return c_name, c_hash

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
        self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return new_columns, new_hashes

    def p_align(self):
        new_columns = []
        new_hashes = []
        current_columns = self.nodes[0].c_name
        current_hashes = self.nodes[0].c_hash
        for i in range(len(current_columns)):
            if current_columns[i] in self.nodes[1].c_name:
                new_columns.append(current_columns[i])
                new_hashes.append(current_hashes[i])

        return new_columns, new_hashes

    # TODO: There may be a better way of hashing these so if the columns are copied and same operations are applied
    # we save storage space
    def p___div__(self):
        c_name = '__div__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__div__',
                                          self.nodes[0].get_materialized_data() / self.nodes[1].get_materialized_data(),
                                          c_name, c_hash)

    def p___rdiv__(self):
        c_name = '__rdiv__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__rdiv__',
                                          self.nodes[1].get_materialized_data() / self.nodes[0].get_materialized_data(),
                                          c_name, c_hash)

    def p___add__(self):
        c_name = '__add__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__add__',
                                          self.nodes[0].get_materialized_data() + self.nodes[1].get_materialized_data(),
                                          c_name, c_hash)

    def p___radd__(self):
        c_name = '__radd__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__radd__',
                                          self.nodes[1].get_materialized_data() + self.nodes[0].get_materialized_data(),
                                          c_name, c_hash)

    def p___sub__(self):
        c_name = '__sub__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__sub__',
                                          self.nodes[0].get_materialized_data() - self.nodes[1].get_materialized_data(),
                                          c_name, c_hash)

    def p___rsub__(self):
        c_name = '__rsub__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__rub__',
                                          self.nodes[1].get_materialized_data() - self.nodes[0].get_materialized_data(),
                                          c_name, c_hash)

    def p___lt__(self):
        c_name = '__lt__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__lt__',
                                          self.nodes[0].get_materialized_data() < self.nodes[1].get_materialized_data(),
                                          c_name, c_hash)

    def p___le__(self):
        c_name = '__le__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__le__', self.nodes[0].get_materialized_data() <= self.nodes[
            1].get_materialized_data(), c_name, c_hash)

    def p___eq__(self):
        c_name = '__eq__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__eq__', self.nodes[0].get_materialized_data() == self.nodes[
            1].get_materialized_data(), c_name, c_hash)

    def p___ne__(self):
        c_name = '__ne__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__ne__', self.nodes[0].get_materialized_data() != self.nodes[
            1].get_materialized_data(), c_name, c_hash)

    def p___qt__(self):
        c_name = '__qt__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__qt__',
                                          self.nodes[0].get_materialized_data() > self.nodes[1].get_materialized_data(),
                                          c_name, c_hash)

    def p___qe__(self):
        c_name = '__qe__'
        c_hash = self.md5(self.generate_uuid())
        return self.hash_and_store_series('__qe__', self.nodes[0].get_materialized_data() >= self.nodes[
            1].get_materialized_data(), c_name, c_hash)


class Feature(Node):
    """ Feature class representing one (and only one) column of a data.
    This class is analogous to pandas.core.series.Series

    Todo:
        * Support for Python 3.x

    """

    # TODO: we really don't need a dictionary for the column_map rather only one key/value pair
    # TODO: is there a better way of representing this? now we must get the index 0 of the values
    # TODO: when constructing the feature from the data storage
    def __init__(self, node_id, execution_environment, c_name='', c_hash='', size=-1):
        Node.__init__(self, node_id, execution_environment)
        assert isinstance(c_name, str)
        assert isinstance(c_hash, str)
        self.c_name = c_name
        self.c_hash = c_hash
        self.size = size

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.optimizer.optimize(
                self.execution_environment.history_graph,
                self.execution_environment.workload_graph,
                self.id,
                verbose)
            self.computed = True
        # TODO: Remove index [0] after the column_map input is changed from dictionary to a better data structure
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.execution_environment.data_storage.get_column(self.c_name, self.c_hash)

    def compute_size(self):
        if self.size == -1:
            self.size = self.execution_environment.data_storage.get_size([self.c_hash])
            return self.size
        else:
            return self.size

    def setname(self, name):
        return self.generate_feature_node('setname', {'name': name})

    def p_setname(self, name):
        return name, self.c_hash

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
        return self.hash_and_store_series('__mul__{}'.format(other), self.get_materialized_data() * other)

    def __rmul__(self, other):
        return self.math('__rmul__', other)

    def p___rmul__(self, other):
        return self.hash_and_store_series('__rmul__{}'.format(other), other * self.get_materialized_data())

    # TODO: When switching to python 3 this has to change to __floordiv__ and __truediv__
    def __div__(self, other):
        return self.math('__div__', other)

    def p___div__(self, other):
        return self.hash_and_store_series('__div__{}'.format(other), self.get_materialized_data() / other)

    def __rdiv__(self, other):
        return self.math('__rdiv__', other)

    def p___rdiv__(self, other):
        return self.hash_and_store_series('__rdiv__{}'.format(other), other / self.get_materialized_data())

    def __add__(self, other):
        return self.math('__add__', other)

    def p___add__(self, other):
        return self.hash_and_store_series('__add__{}'.format(other), self.get_materialized_data() + other)

    def __radd__(self, other):
        return self.math('__radd__', other)

    def p___radd__(self, other):
        return self.hash_and_store_series('__radd__{}'.format(other), other + self.get_materialized_data())

    def __sub__(self, other):
        return self.math('__sub__', other)

    def p___sub__(self, other):
        return self.hash_and_store_series('__sub_{}'.format(other), self.get_materialized_data() - other)

    def __rsub__(self, other):
        return self.math('__rsub__', other)

    def p___rsub__(self, other):
        return self.hash_and_store_series('__rsub__{}'.format(other), other - self.get_materialized_data())

    def __lt__(self, other):
        return self.math('__lt__', other)

    def p___lt__(self, other):
        return self.hash_and_store_series('__lt__{}'.format(other), self.get_materialized_data() < other)

    def __le__(self, other):
        return self.math('__le__', other)

    def p___le__(self, other):
        return self.hash_and_store_series('__le__{}'.format(other), self.get_materialized_data() <= other)

    def __eq__(self, other):
        return self.math('__eq__', other)

    def p___eq__(self, other):
        return self.hash_and_store_series('__eq__{}'.format(other), self.get_materialized_data() == other)

    def __ne__(self, other):
        return self.math('__ne__', other)

    def p___ne__(self, other):
        return self.hash_and_store_series('__ne__{}'.format(other), self.get_materialized_data() != other)

    def __gt__(self, other):
        return self.math('__qt__', other)

    def p___gt__(self, other):
        return self.hash_and_store_series('__gt__{}'.format(other), self.get_materialized_data() > other)

    def __ge__(self, other):
        return self.math('__qe__', other)

    def p___ge__(self, other):
        return self.hash_and_store_series('__ge__{}'.format(other), self.get_materialized_data() >= other)

    # End of overridden methods

    def head(self, size=5):
        return self.generate_feature_node('head', {'size': size})

    def p_head(self, size=5):
        return self.hash_and_store_series('head{}'.format(size), self.get_materialized_data().head(size))

    def fillna(self, value):
        return self.generate_feature_node('fillna', {'value': value})

    def p_fillna(self, value):
        return self.hash_and_store_series('fill{}'.format(value), self.get_materialized_data().fillna(value))

    # combined node
    def concat(self, nodes):
        if type(nodes) == list:
            supernode = self.generate_super_node([self] + nodes, {'c_oper': 'concat'})
        else:
            supernode = self.generate_super_node([self] + [nodes], {'c_oper': 'concat'})
        return self.generate_dataset_node('concat', v_id=supernode.id)

    def isnull(self):
        self.execution_environment.workload_graph.add_edge(self.id,
                                                           {'oper': self.e_hash('isnull'),
                                                            'hash': self.e_hash('isnull')},
                                                           ntype=type(self).__name__)

    def notna(self):
        return self.generate_feature_node('notna')

    def p_notna(self):
        return self.hash_and_store_series('notna', self.get_materialized_data().notna())

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
        return self.hash_and_store_series('abs', self.get_materialized_data().abs())

    def unique(self):
        return self.generate_feature_node('unique')

    def p_unique(self):
        return self.hash_and_store_series('unique', self.get_materialized_data().unique())

    def dropna(self):
        return self.generate_feature_node('dropna')

    def p_dropna(self):
        return self.hash_and_store_series('dropna', self.get_materialized_data().dropna())

    def binning(self, start_value, end_value, num):
        return self.generate_feature_node('binning',
                                          {'start_value': start_value, 'end_value': end_value, 'num': num})

    def p_binning(self, start_value, end_value, num):
        return self.hash_and_store_series('binning{}{}{}'.format(start_value, end_value, num),
                                          pd.cut(self.get_materialized_data(),
                                                 bins=np.linspace(start_value, end_value, num=num)))

    def replace(self, to_replace):
        return self.generate_feature_node('replace', {'to_replace': to_replace})

    def p_replace(self, to_replace):
        return self.hash_and_store_series('__replace__{}'.format(str(to_replace)),
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

    def p_fit_sk_model(self, model):
        start = datetime.now()
        model.fit(self.get_materialized_data())
        self.execution_environment.update_time('model-training', (datetime.now() - start).total_seconds())
        return model


class Dataset(Node):
    """ Dataset class representing a dataset (set of Features)
    This class is analogous to pandas.core.frame.DataFrame

    TODO:
        * Integration with the graph library
        * Add support for every experiment_graph that Pandas DataFrame supports
        * Support for Python 3.x

    """

    def __init__(self, node_id, execution_environment, c_name=None, c_hash=None, size=-1):
        Node.__init__(self, node_id, execution_environment)
        self.c_name = [] if c_name is None else c_name
        self.c_hash = [] if c_hash is None else c_hash
        self.size = size

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.optimizer.optimize(
                self.execution_environment.history_graph,
                self.execution_environment.workload_graph,
                self.id,
                verbose)

            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.execution_environment.data_storage.get_dataset(self.c_name, self.c_hash)

    def compute_size(self):
        if self.size == -1:
            self.size = self.execution_environment.data_storage.get_size(self.c_hash)
            return self.size
        else:
            return self.size

    def set_columns(self, columns):
        return self.generate_dataset_node('set_columns', {'columns': columns})

    def p_set_columns(self, columns):
        df = self.get_materialized_data()
        self.execution_environment.data_storage.store_dataset(self.c_hash, df)
        return columns, self.c_hash

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
                self.execution_environment.data_storage.store_dataset(p_hashes, self.get_materialized_data()[p_columns])
            return p_columns, p_hashes
        else:
            self.execution_environment.data_storage.store_column(self.get_c_hash(columns),
                                                                 self.get_materialized_data()[columns])
            return columns, self.get_c_hash(columns)

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

    def p_reset_index(self):
        df = self.get_materialized_data().reset_index()
        new_column = df.columns[0]
        new_column_data = df.iloc[0]
        new_c_hash = self.md5(self.generate_uuid())
        self.execution_environment.data_storage.store_column(new_c_hash, new_column_data)
        del df
        del new_column_data
        gc.collect()
        return [new_column] + self.c_name, [new_c_hash] + self.c_hash

    def copy(self):
        return self.generate_dataset_node('copy', c_name=self.c_name, c_hash=self.c_hash)

    def p_copy(self):
        return self.c_name, self.c_hash

    def head(self, size=5):
        return self.generate_dataset_node('head', {'size': size})

    def p_head(self, size=5):
        return self.hash_and_store_df('head{}'.format(size), self.get_materialized_data().head(size))

    def shape(self):
        return self.generate_agg_node('shape', {})

    def p_shape(self):
        return self.get_materialized_data().shape

    def isnull(self):
        return self.generate_dataset_node('isnull')

    def p_isnull(self):
        return self.hash_and_store_df('isnull', self.get_materialized_data().isnull())

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
        return self.hash_and_store_df('abs', self.get_materialized_data().abs())

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
        return self.hash_and_store_df('notna', self.get_materialized_data().notna())

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
        self.execution_environment.data_storage.store_dataset(c_hashes, df[c_names])
        return c_names, c_hashes

    # If drop column results in one column the return type should be a Feature
    def drop(self, columns):
        return self.generate_dataset_node('drop', {'columns': columns})

    def p_drop(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        new_c = []
        new_hash = []
        for c in self.c_name:
            if c not in columns:
                new_c.append(c)
                new_hash.append(self.get_c_hash(c))
        self.execution_environment.data_storage.store_dataset(new_hash, self.get_materialized_data()[new_c])
        return new_c, new_hash
        # return self.hash_and_store_df(self.map, 'drop{}'.format(columns), self.get_materialized_data().drop(columns=columns))

    def dropna(self):
        return self.generate_dataset_node('dropna')

    def p_dropna(self):
        return self.hash_and_store_df('dropna', self.get_materialized_data().dropna())

    def sort_values(self, col_name, ascending=False):
        return self.generate_dataset_node('sort_values', args={'col_name': col_name, 'ascending': ascending})

    def p_sort_values(self, col_name, ascending):
        return self.hash_and_store_df('sort_values{}{}'.format(col_name, ascending),
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
            if c in self.c_name:
                new_hash.append(self.get_c_hash(c))
            else:
                # generate a unique prefix to make sure onehot encoding of different datasets with the same
                # column name does not result in the same column hash
                new_hash.append(self.md5(self.generate_uuid() + c))

        self.execution_environment.data_storage.store_dataset(new_hash, df[new_column])
        return new_column, new_hash

    def corr(self):
        return self.generate_agg_node('corr', {})

    def p_corr(self):
        return self.get_materialized_data().corr()

    # TODO: is it OK to assume as_index is always false?
    # we can recreate the original one anyway
    # For now we materialize the result as an aggregate node so everything will be stored inside the graph
    def groupby(self, col_names):
        return self.generate_groupby_node('groupby', {'col_names': col_names, 'as_index': False})

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
        for c in self.c_name:
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
        supernode = self.generate_super_node([self, other], {'c_oper': 'align'})
        return self.generate_dataset_node('align', v_id=supernode.id)

    # dataframe merge operation operation of dataframes
    def merge(self, other, on, how='left'):
        supernode = self.generate_super_node([self, other], {'c_oper': 'merge'})
        return self.generate_dataset_node('merge', args={'on': on, 'how': how}, v_id=supernode.id)

    def fit_sk_model(self, model):
        return self.generate_sklearn_node('fit_sk_model', {'model': model})

    def fit_sk_model_with_labels(self, model, labels, custom_args=None):
        supernode = self.generate_super_node([self, labels], {'c_oper': 'fit_sk_model_with_labels'})
        return self.generate_sklearn_node('fit_sk_model_with_labels', {'model': model, 'custom_args': custom_args},
                                          v_id=supernode.id)

    def p_fit_sk_model(self, model):
        start = datetime.now()
        model.fit(self.get_materialized_data())
        self.execution_environment.update_time('model-training', (datetime.now() - start).total_seconds())
        return model

    def replace_columns(self, col_names, features):
        if type(features) == list:
            supernode = self.generate_super_node([self] + features,
                                                 {'col_names': col_names, 'c_oper': 'replace_columns'})
        else:
            supernode = self.generate_super_node([self, features],
                                                 {'col_names': col_names, 'c_oper': 'replace_columns'})

        return self.generate_dataset_node('replace_columns', {'col_names': col_names}, v_id=supernode.id)


# TODO: for now we are always generating randomized hashes for the generated aggregate columns
# However, there should be a better way since we are losing some deduplication opportunities
# e.g., if user performs groupby().count() on the same columns on two datasets, the results will be
# duplicated and stored twice with different hashes
# We should find a way to also store groupby graph inside the data storage
# this way we can generate consistent hashes
class GroupBy(Node):
    def __init__(self, node_id, execution_environment, data_obj=None):
        Node.__init__(self, execution_environment, node_id)
        self.data_obj = data_obj

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.optimizer.optimize(
                self.execution_environment.history_graph,
                self.execution_environment.workload_graph,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.data_obj

    def project(self, columns):
        return self.generate_groupby_node('project', {'columns': columns})

    def p_project(self, columns):
        data_object = self.get_materialized_data()
        df = data_object[0]
        key_columns = copy.deepcopy(data_object[1])
        key_hashes = copy.deepcopy(data_object[2])
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
        new_columns = copy.deepcopy(data_object[1])
        new_hashes = copy.deepcopy(data_object[2])

        c_group_names = data_object[3]
        c_group_hashes = data_object[4]

        # find the selected subset
        for i in range(len(c_group_names)):
            new_columns.append(c_group_names[i])
            new_hashes.append(self.md5(c_group_hashes[i] + 'count'))
        self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return new_columns, new_hashes

    def agg(self, functions):
        return self.generate_dataset_node('agg', {'functions': functions})

    def p_agg(self, functions):
        data_object = self.get_materialized_data()
        # .reset_index() is essential as groupby (as_index=False) has no effect for agg function
        df = self.get_materialized_data()[0].agg(functions).reset_index()
        new_columns = copy.deepcopy(data_object[1])
        new_hashes = copy.deepcopy(data_object[2])

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
        self.execution_environment.data_storage.store_dataset(new_hashes, df)
        return new_columns, new_hashes

    def mean(self):
        return self.generate_dataset_node('mean')

    def p_mean(self):
        data_object = self.get_materialized_data()
        df = data_object[0].mean()
        new_columns = copy.deepcopy(data_object[1])
        new_hashes = copy.deepcopy(data_object[2])

        c_group_names = data_object[3]
        c_group_hashes = data_object[4]

        # find the selected subset
        for i in range(len(c_group_names)):
            new_columns.append(c_group_names[i])
            new_hashes.append(self.md5(c_group_hashes[i] + 'mean'))

        self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return new_columns, new_hashes


class Agg(Node):
    def __init__(self, node_id, execution_environment, data_obj=None):
        Node.__init__(self, node_id, execution_environment)
        self.data_obj = data_obj

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.optimizer.optimize(
                self.execution_environment.history_graph,
                self.execution_environment.workload_graph,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.data_obj

    def show(self):
        return self.id + " :" + self.get_materialized_data().__str__()


class SK_Model(Node):
    def __init__(self, node_id, execution_environment, data_obj=None):
        Node.__init__(self, node_id, execution_environment)
        self.data_obj = data_obj

    def data(self, verbose=0):
        self.update_freq()
        if not self.computed:
            self.execution_environment.optimizer.optimize(
                self.execution_environment.history_graph,
                self.execution_environment.workload_graph,
                self.id,
                verbose)
            self.computed = True
        return self.get_materialized_data()

    def get_materialized_data(self):
        return self.data_obj

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

        self.execution_environment.data_storage.store_dataset(new_hashes, df[new_columns])
        return new_columns, new_hashes

    def predict_proba(self, test, custom_args=None):
        supernode = self.generate_super_node([self, test], {'c_oper': 'predict_proba'})
        return self.generate_dataset_node('predict_proba', args={'custom_args': custom_args}, v_id=supernode.id)
