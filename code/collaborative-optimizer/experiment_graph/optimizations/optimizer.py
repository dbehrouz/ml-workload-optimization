"""
The optimizer module performs the scheduling.
It receives the workload execution graph and the historical execution graph and optimizes
the workload execution graph and returns the scheduled execution path
"""
import copy
from abc import abstractmethod
from datetime import datetime

from Reuse import Reuse


class Optimizer:
    NAME = 'BASE_OPTIMIZER'

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
    def optimize(self, history, workload, v_id, verbose):
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
    def copy_from_history(history_node, workload_node):
        workload_node['data'].computed = True
        workload_node['size'] = history_node['size']
        if history_node['type'] == 'Dataset' or history_node['type'] == 'Feature':
            workload_node['data'].c_name, workload_node['data'].c_hash = copy.copy((
                history_node['data'].c_name, history_node['data'].c_hash))
        else:
            # TODO we should make sure shallow copy is not problematic for groupby operations
            workload_node['data'].data_obj = copy.deepcopy(history_node['data'].data_obj)

    @staticmethod
    def get_optimizer(optimizer_type, reuse_type):
        optimizer_type = optimizer_type.upper()
        if optimizer_type == HashBasedOptimizer.NAME:
            return HashBasedOptimizer(reuse_type)
        else:
            raise Exception('Unknown Optimizer type: {}'.format(optimizer_type))

    def reuse(self):
        return Reuse.get_reuse(self.reuse_type)


# class SearchBasedOptimizer(Optimizer):
#     NAME = 'SEARCH_BASED'
#
#     def optimize(self, history, workload, v_id, verbose=0):
#         """
#         receives the historical graph and the workload graph and optimizes workload graph
#         and returns a optimized graph containing the required materialized artifact from
#         the history graph. Here are the steps:
#         1. find the optimized subgraph of the workload
#         2. optimize the workload subgraph further using the history
#
#         :param verbose:
#         :param v_id:
#         :param history:
#         :param workload:
#         :return:
#         """
#         start = datetime.now()
#         workload_subgraph = workload.compute_execution_subgraph(v_id)
#         if verbose == 1:
#             print 'workload graph size: {}'.format(len(workload_subgraph.nodes()))
#             print 'materialized nodes before optimization begins: {}'.format(self.materialized_nodes)
#         # here we optimize workload graph with historical graph
#         if not history.is_empty():
#             reuse_optimization_time_start = datetime.now()
#             self.cross_optimize(history, workload, workload_subgraph, verbose)
#             workload_subgraph = workload.compute_execution_subgraph(v_id)
#             reuse_optimization = (datetime.now() - reuse_optimization_time_start).total_seconds()
#         else:
#             reuse_optimization = 0
#
#         # print 'size of the optimized graph = {}'.format(len(optimized_subgraph))
#         if verbose == 1:
#             print 'optimized workload graph size: {}'.format(len(workload_subgraph.nodes()))
#
#         # execute the workload using the optimized view
#         final_schedule = workload.compute_result_with_subgraph(workload_subgraph, verbose)
#
#         lapsed = (datetime.now() - start).total_seconds()
#         if v_id in self.times:
#             raise Exception('something is wrong, {} should have been already computed'.format(v_id))
#         else:
#             # TODO only in debug mode this should be reported.
#             # Once we have a Logger module we can report this sa well
#             # path = ''
#             # for pair in final_schedule:
#             #     if path == '':
#             #         path = pair[0]
#             #     path += '-' + workload.graph.edges[pair[0], pair[1]]['name'] + '->' + pair[1]
#             self.times[v_id] = (lapsed - reuse_optimization, reuse_optimization)
#
#     @staticmethod
#     def extend(history, workload):
#         history.extend(workload)
#
#     def cross_optimize(self, history, workload, workload_subgraph, verbose):
#         """
#         also set the data node of the workload graph from the history graph
#         :param verbose:
#         :param workload_subgraph:
#         :param history:
#         :param workload:
#         :return:
#         """
#         self.find_materialized_nodes(history, workload, workload_subgraph, verbose)
#
#     def find_materialized_nodes(self, history, workload, workload_subgraph, verbose):
#         """
#         TODO this is a bit clunky, at some point we should try to do a better design
#         TODO so we don't have to mutate the workload graph inside this function
#
#         find nodes in history graph that can the subgraph view can utilize
#         and then copy their data to the workload graph
#         :param verbose:
#         :param history:
#         :param workload:
#         :param workload_subgraph:
#         :return:
#         """
#         # find root nodes of the workload subgraph
#         # important precondition, root vertices always have unique ids  therefore if the history graph contains a root,
#         # the id should match with the one in the workload graph
#         roots = [(v, v) for v, d in workload_subgraph.nodes(data='root') if d and history.has_node(v)]
#         materialized_nodes = {}
#         if verbose == 1:
#             print 'existing materialized nodes {}'.format(self.materialized_nodes)
#         for w, h in self.materialized_nodes.iteritems():
#             roots.append((w, h))
#             materialized_nodes[w] = h
#
#         for r in roots:
#             # materialized_nodes are the set of nodes that exists in both graphs and are materialized
#             # in the history graph
#             self.find_furthest_materialized_nodes(history.graph, workload_subgraph, r, materialized_nodes)
#         if verbose == 1:
#             print 'new materialized nodes {}'.format(materialized_nodes)
#         for m_w, m_h in materialized_nodes.iteritems():
#             history_node = history.graph.nodes[m_h]
#             workload_node = workload.graph.nodes[m_w]
#             self.copy_from_history(history_node, workload_node)
#             if m_w in self.materialized_nodes:
#                 if self.materialized_nodes[m_w] != m_h:
#                     raise Exception(
#                         'the value of the key \'{}\'is changing, existing value \'{}\', new value \'{}\''.format(
#                             m_w, self.materialized_nodes[m_w], m_h, ))
#             else:
#                 self.materialized_nodes[m_w] = m_h
#
#     @staticmethod
#     def find_furthest_materialized_nodes(history_graph, workload_graph, root_vertex_pair, materialized_nodes):
#         """
#         given a root vertex and a source graph, find the set of materialized nodes in the target_graph
#         that are furthest from the source and exist in both the source and target graph
#         the root in the graph and returns vertex ids
#         :param materialized_nodes: initial materialized nodes
#         :param history_graph: history graph
#         :param workload_graph: workload sub graph
#         :param root_vertex_pair: tuple of the form (w_root, h_root) a mapping of root and materialized nodes between
#             the two graphs
#         :return:
#         """
#         valid_nodes = [root_vertex_pair]
#         # materialized_nodes = {root_vertex_pair[0]: root_vertex_pair[1]}
#
#         out_degree = {v: d for v, d in workload_graph.out_degree() if d > 0}
#         combined_nodes = {}
#         potential_valid_super_nodes = {}
#         while valid_nodes:
#             while valid_nodes:
#                 w, h = valid_nodes.pop()
#                 # w_e is a tuple of (source, destination, data_dictionary)
#                 for _, w_destination, w_edge_data in workload_graph.out_edges(w, data=True):
#                     # if the edge is a combine
#                     if w_edge_data['name'] == COMBINE_OPERATION_IDENTIFIER:
#                         if w_destination in combined_nodes:
#                             combined_nodes[w_destination].add((w, h))
#                         else:
#                             combined_nodes[w_destination] = {(w, h)}
#                         if {wn for wn, hn in combined_nodes[w_destination]} == set(
#                                 workload_graph.node[w_destination]['involved_nodes']):
#                             potential_valid_super_nodes[w_destination] = copy.deepcopy(
#                                 combined_nodes[w_destination])
#                     else:
#                         for _, h_destination, h_edge_data in history_graph.out_edges(h, data=True):
#                             if w_edge_data['hash'] == h_edge_data['hash']:
#                                 if w_destination not in materialized_nodes:
#                                     valid_nodes.append((w_destination, h_destination))
#                                     if history_graph.nodes[h_destination]['data'].computed:
#                                         materialized_nodes[w_destination] = h_destination
#                                         out_degree[w] -= 1
#
#                 # TODO when a node's out_degree becomes 0, we should also decrease its parent's out_degree by 1 and
#                 # TODO check if that should be remove or not and so on ... until we reach parent node with out_degree
#                 # TODO > 1 or root
#                 if w in out_degree and out_degree[w] == 0:
#                     if w in materialized_nodes:
#                         del materialized_nodes[w]
#
#             for super_node, involved in potential_valid_super_nodes.iteritems():
#                 h_super_node = None
#                 for w, h in involved:
#                     add_to_valid = False
#                     w_hash = workload_graph[w][super_node]['hash']
#                     for _, h_destination, h_hash in history_graph.out_edges(h, data='hash'):
#                         if h_hash == w_hash:
#                             if h_super_node is not None:
#                                 assert h_super_node == h_destination
#                             else:
#                                 h_super_node = h_destination
#                             add_to_valid = True
#                     if not add_to_valid:
#                         break
#
#                 h_node = list(history_graph.successors(h_super_node))
#                 w_node = list(workload_graph.successors(super_node))
#                 assert len(h_node) == 1
#                 assert len(w_node) == 1
#                 h_node = h_node[0]
#                 w_node = w_node[0]
#                 if w_node not in materialized_nodes:
#                     valid_nodes.append((w_node, h_node))
#                     if history_graph.nodes[h_node]['data'].computed:
#                         materialized_nodes[w_node] = h_node
#                         for w, _ in involved:
#                             out_degree[w] -= 1
#                             if out_degree[w] == 0:
#                                 if w in materialized_nodes:
#                                     del materialized_nodes[w]
#                 else:
#                     for w, _ in involved:
#                         out_degree[w] -= 1
#                         if out_degree[w] == 0:
#                             if w in materialized_nodes:
#                                 del materialized_nodes[w]
#
#             potential_valid_super_nodes = {}
#
#         return materialized_nodes


class HashBasedOptimizer(Optimizer):
    NAME = 'HASH_BASED'

    def optimize(self, history, workload, v_id, verbose):
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
        materialized_vertices, execution_vertices, total_history_graph_reads = self.reuse().run(vertex=vertex,
                                                                                                workload=workload,
                                                                                                history=history,
                                                                                                verbose=verbose)
        self.history_reads += total_history_graph_reads
        for m in materialized_vertices:
            self.copy_from_history(history.graph.nodes[m], workload.graph.nodes[m])
        print 'terminal: {}, materialized: {}, execution: {}, reads: {}'.format(vertex, materialized_vertices,
                                                                                execution_vertices,
                                                                                total_history_graph_reads)
        return workload.graph.subgraph(execution_vertices)
