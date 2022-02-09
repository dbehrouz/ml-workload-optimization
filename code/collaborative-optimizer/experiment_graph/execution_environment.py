import os
import pickle

from experiment_graph.data_storage import SimpleStorageManager
from experiment_graph.graph.graph_representations import WorkloadDag, ExperimentGraph
# Reserved word for representing super graph.
# Do not use combine as an operation name
from experiment_graph.graph.node import *
from experiment_graph.optimizations.Reuse import LinearTimeReuse
from experiment_graph.optimizations.collaborativescheduler import HashBasedCollaborativeScheduler, \
    CollaborativeScheduler


class ExecutionEnvironment(object):

    @staticmethod
    def construct_readable_root_hash(loc, extra_params=None):
        return loc[loc.rfind('/') + 1:] + str(extra_params)

    def __init__(self, data_storage=SimpleStorageManager(), scheduler_type=HashBasedCollaborativeScheduler.NAME,
                 reuse_type=LinearTimeReuse.NAME):
        self.scheduler = CollaborativeScheduler.get_scheduler(scheduler_type, reuse_type)
        self.workload_dag = WorkloadDag()
        self.experiment_graph = ExperimentGraph(data_storage=data_storage)
        self.time_manager = dict()

    def get_benchmark_results(self, keys=None):
        if BenchmarkMetrics.TOTAL_EXECUTION not in self.time_manager:
            self.compute_total_reuse_optimization_time()
        if keys is None:
            return ','.join(
                ['NOT CAPTURED' if key not in self.time_manager else str(self.time_manager[key]) for key in
                 BenchmarkMetrics.keys])
        else:
            return ','.join([self.time_manager[key] for key in keys])

    def update_time(self, oper_type, seconds):
        if oper_type in self.time_manager:
            self.time_manager[oper_type] = self.time_manager[oper_type] + seconds
        else:
            self.time_manager[oper_type] = seconds

    def update_history(self):
        start = datetime.now()
        self.experiment_graph.extend(self.workload_dag)
        self.update_time(BenchmarkMetrics.UPDATE_HISTORY, (datetime.now() - start).total_seconds())

    def mock_update_history(self):
        self.experiment_graph.mock_extend(self.workload_dag)

    def save_history(self, environment_folder, overwrite=False, skip_history_update=False):
        if os.path.exists(environment_folder) and not overwrite:
            raise Exception('Directory already exists and overwrite is not allowed')
        if not os.path.exists(environment_folder):
            os.makedirs(environment_folder)
        start_save_graph = datetime.now()

        with open(environment_folder + '/graph', 'wb') as output:
            pickle.dump(self.experiment_graph.graph, output, pickle.HIGHEST_PROTOCOL)

        with open(environment_folder + '/roots', 'wb') as output:
            pickle.dump(self.experiment_graph.roots, output, pickle.HIGHEST_PROTOCOL)

        end_save_graph = datetime.now()

        self.update_time(BenchmarkMetrics.SAVE_HISTORY, (end_save_graph - start_save_graph).total_seconds())
        with open(environment_folder + '/storage', 'wb') as output:
            pickle.dump(self.experiment_graph.data_storage, output, pickle.HIGHEST_PROTOCOL)

        self.update_time(BenchmarkMetrics.SAVE_DATA_STORE, (datetime.now() - end_save_graph).total_seconds())

    def compute_total_reuse_optimization_time(self):
        # optimizer.times has  the form {vertex_id:(execution time, optimization time)}
        total_execution_time = 0
        total_reuse_time = 0
        for _, v in self.scheduler.times.items():
            total_execution_time += v[0]
            total_reuse_time += v[1]
        self.update_time(BenchmarkMetrics.TOTAL_EXECUTION, total_execution_time)
        self.update_time(BenchmarkMetrics.TOTAL_REUSE, total_reuse_time)
        self.update_time(BenchmarkMetrics.TOTAL_HISTORY_READ, self.scheduler.history_reads)

    def new_workload(self):
        """
        call this function if you want to keep the history graph and start a new workload in the same execution
        environment
        :return:
        """
        del self.workload_dag
        self.workload_dag = WorkloadDag()
        del self.time_manager
        self.time_manager = dict()
        scheduler_type = self.scheduler.NAME
        reuse_type = self.scheduler.reuse_type
        del self.scheduler
        self.scheduler = CollaborativeScheduler.get_scheduler(scheduler_type, reuse_type)

    def load_history_from_memory(self, history):
        self.experiment_graph = history

    def load_history_from_disk(self, environment_folder):
        start_graph_load = datetime.now()
        if not os.path.isdir(environment_folder):
            os.makedirs(environment_folder)
        with open(environment_folder + '/graph', 'rb') as g_input:
            graph = pickle.load(g_input)

        with open(environment_folder + '/roots', 'rb') as g_input:
            roots = pickle.load(g_input)

        with open(environment_folder + '/storage', 'rb') as d_input:
            data_storage = pickle.load(d_input)
        self.experiment_graph = ExperimentGraph(data_storage, graph, roots)

        self.update_time(BenchmarkMetrics.LOAD_HISTORY, (datetime.now() - start_graph_load).total_seconds())

    def plot_graph(self, plt, graph_type='workload'):
        if graph_type == 'workload':
            self.workload_dag.plot_graph(plt)
        else:
            self.experiment_graph.plot_graph(plt)

    def get_artifacts_size(self, graph_type='workload'):
        if graph_type == 'workload':
            return self.workload_dag.get_artifact_sizes()
        else:
            return self.experiment_graph.get_artifact_sizes()

    def load(self, loc, dtype=None, nrows=None, parse_dates=False, date_parser=None):
        extra_params = dict()
        if dtype is not None:
            extra_params['dtype'] = dtype
        if nrows is not None:
            extra_params['nrows'] = nrows
        if not parse_dates:
            extra_params['parse_dates'] = parse_dates
        if date_parser is not None:
            extra_params['date_parser'] = date_parser

        root_hash = self.construct_readable_root_hash(loc, extra_params)
        if self.workload_dag.has_node(root_hash):
            # print 'loading root node {} from workload graph'.format(root_hash)
            return self.workload_dag.get_node(root_hash)['data']
        elif self.experiment_graph.has_node(root_hash):
            # print 'loading root node {} from history graph'.format(root_hash)
            root = copy.deepcopy(self.experiment_graph.graph.nodes[root_hash])
            root['data'].execution_environment = self
            root['data'].underlying_data = self.experiment_graph.retrieve_data(root_hash)
            self.workload_dag.roots.append(root_hash)
            self.workload_dag.add_node(root_hash, **root)
            return root['data']
        else:
            print('creating a new root node')
            start = datetime.now()
            initial_data = pd.read_csv(loc, dtype=dtype, nrows=nrows, parse_dates=parse_dates, date_parser=date_parser)
            end = datetime.now()
            self.update_time(BenchmarkMetrics.LOAD_DATASET, (end - start).total_seconds())
            c_name = []
            c_hash = []
            # create the md5 hash values for columns
            for i in range(len(initial_data.columns)):
                c = initial_data.columns[i]
                c_name.append(c)
                # to ensure the same hashing is used for the initial data loading and the rest of the artifacts
                # Adding the loc to make sure different datasets with the same column names do not mix
                c_hash.append(Node.md5(root_hash + c))

            # self.data_storage.store_dataset(c_hash, initial_data[c_name])
            df = DataFrame(column_names=c_name, column_hashes=c_hash, pandas_df=initial_data[c_name])
            nextnode = Dataset(root_hash, self, underlying_data=df)
            node_size_start = datetime.now()
            # size = nextnode.compute_size()
            self.update_time(BenchmarkMetrics.NODE_SIZE_COMPUTATION,
                             (datetime.now() - node_size_start).total_seconds())
            self.workload_dag.roots.append(root_hash)
            self.workload_dag.add_node(root_hash, **{'root': True, 'type': 'Dataset', 'data': nextnode,
                                                     'loc': loc,
                                                     'extra_params': extra_params,
                                                     'size': None})
            return nextnode

    def load_from_pandas(self, df, identifier):
        if self.workload_dag.has_node(identifier):
            return self.workload_dag.get_node(identifier)['data']
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

            # self.data_storage.store_dataset(c_hash, df[c_name])
            df = DataFrame(column_names=c_name, column_hashes=c_hash, pandas_df=df[c_name])
            nextnode = Dataset(identifier, self, underlying_data=df)
            nextnode.computed = True
            node_size_start = datetime.now()
            # size = nextnode.compute_size()
            self.update_time(BenchmarkMetrics.NODE_SIZE_COMPUTATION,
                             (datetime.now() - node_size_start).total_seconds())
            self.workload_dag.roots.append(identifier)
            self.workload_dag.add_node(identifier,
                                       **{'root': True, 'type': 'Dataset', 'data': nextnode,
                                          'loc': identifier,
                                          'extra_params': 'load_from_memory',
                                          'size': None})
            return nextnode

    def empty_node(self, node_type='Dataset', identifier='empty_root'):
        if self.workload_dag.has_node(identifier):
            return self.workload_dag.get_node(identifier)['data']

        if node_type == 'Dataset':
            return Dataset(identifier, self)
        elif node_type == 'Feature':
            return Feature(identifier, self)
        else:
            raise TypeError(f'Unknown Data Type: {node_type}')
