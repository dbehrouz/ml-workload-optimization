"""
The CollaborativeScheduler module performs the scheduling and Reuse
It receives the workload execution graph and the historical execution graph and a reuse algorirthm and
optimizes the workload execution graph and returns the scheduled execution path
"""
import copy
from abc import abstractmethod
from datetime import datetime

from Reuse import Reuse
from experiment_graph.graph.graph_representations import WorkloadDag
from experiment_graph.graph.graph_representations import ExperimentGraph


class CollaborativeScheduler:
    NAME = 'BASE_SCHEDULER'

    def __init__(self, reuse_type='fast-bottomup'):
        # dictionary for storing pair of materialized nodes between the workload and history graph
        self.materialized_nodes = {}
        # of the form {vertex: (execution_time, optimization_time)
        # if history graph is empty, optimization_time is zero
        # sum of execution_time + optimization_time is total time spent on returning the data in vertex
        # back to the user
        self.times = {}
        self.history_reads = 0
        self.reuse_type = reuse_type

    @abstractmethod
    def schedule(self, history, workload, v_id, verbose):
        """
        receives the historical graph and the workload graph and optimizes workload graph
        and returns a optimized graph containing the required materialized artifact from
        the history graph. Here are the steps:
        :param history:
        :param workload:
        :param v_id:
        :param verbose:
        :return:
        """
        pass

    @staticmethod
    def retrieve_from_history(workload_dag, experiment_graph, node_id):
        """
        :type workload_dag: WorkloadDag
        :type experiment_graph: ExperimentGraph
        :type node_id: str

        """
        underlying_data = experiment_graph.retrieve_data(node_id)
        workload_node = workload_dag.graph.nodes[node_id]
        workload_node['data'].computed = True
        workload_node['size'] = experiment_graph.graph[node_id]['size']
        workload_node['data'].underlying_data = underlying_data

    @staticmethod
    def get_scheduler(optimizer_type, reuse_type):
        optimizer_type = optimizer_type.upper()
        if optimizer_type == HashBasedCollaborativeScheduler.NAME:
            return HashBasedCollaborativeScheduler(reuse_type)
        else:
            raise Exception('Unknown Optimizer type: {}'.format(optimizer_type))

    def reuse(self):
        return Reuse.get_reuse(self.reuse_type)


class HashBasedCollaborativeScheduler(CollaborativeScheduler):
    NAME = 'HASH_BASED'

    def schedule(self, history, workload, v_id, verbose):
        start = datetime.now()
        if not history.is_empty():
            reuse_optimization_time_start = datetime.now()
            workload_subgraph = self.compute_execution_subgraph(history, workload, v_id, verbose=verbose)
            reuse_optimization = (datetime.now() - reuse_optimization_time_start).total_seconds()
        else:
            workload_subgraph = workload.compute_execution_subgraph(v_id)
            reuse_optimization = 0

        final_schedule = workload.compute_result_with_subgraph(workload_subgraph)
        if verbose == 1:
            schedule_length = len(final_schedule)
            if schedule_length > 0:
                print 'executing {} steps to compute vertex {}'.format(schedule_length, v_id)
        lapsed = (datetime.now() - start).total_seconds()
        if v_id in self.times:
            raise Exception('something is wrong, {} should have been already computed'.format(v_id))
        # else:
        #     path = ''
        #     for pair in final_schedule:
        #         if path == '':
        #             path = pair[0]
        #         path += '-' + workload.graph.edges[pair[0], pair[1]]['name'] + '->' + pair[1]
        #     if len(final_schedule) > 0:
        #         print path
        self.times[v_id] = (lapsed - reuse_optimization, reuse_optimization)

    # TODO measure number of reads in history graph
    # TODO measure time of these
    def compute_execution_subgraph(self, history, workload, vertex, verbose):

        materialized_vertices, execution_vertices, warmstarting_candidates, total_history_graph_reads = \
            self.reuse().run(
                vertex=vertex,
                workload=workload,
                history=history,
                verbose=verbose)
        self.history_reads += total_history_graph_reads
        for m in materialized_vertices:
            self.retrieve_from_history(workload, history, m)

        for source, destination, model in warmstarting_candidates:
            workload.graph.edges[source, destination]['args']['model'] = model
            workload.graph.edges[source, destination]['args']['warm_start'] = True

        return workload.graph.subgraph(execution_vertices)
