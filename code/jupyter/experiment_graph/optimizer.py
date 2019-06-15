"""
The optimizer module performs the scheduling.
It receives the workload execution graph and the historical execution graph and optimizes
the workload execution graph and returns the scheduled execution path
"""
import copy

from graph.execution_graph import COMBINE_OPERATION_IDENTIFIER


class Optimizer:

    def __init__(self):
        pass

    def optimize(self, history, workload, v_id, verbose=0):
        """
        receives the historical graph and the workload graph and optimizes workload graph
        and returns a optimized graph containing the required materialized artifact from
        the history graph. Here are the steps:
        1. find the optimized subgraph of the workload
        2. optimize the workload subgraph further using the history

        :param verbose:
        :param v_id:
        :param history:
        :param workload:
        :return:
        """
        workload_subgraph = workload.compute_execution_subgraph(v_id)
        if verbose == 1:
            print 'workload graph size: {}'.format(len(workload_subgraph.nodes()))
        # here we optimize workload graph with historical graph
        if not history.is_empty():
            self.cross_optimize(history, workload, workload_subgraph, verbose)

        optimized_subgraph = workload.compute_execution_subgraph(v_id)

        # print 'size of the optimized graph = {}'.format(len(optimized_subgraph))
        if verbose == 1:
            print 'optimized workload graph size: {}'.format(len(optimized_subgraph.nodes()))

        # execute the workload using the optimized view
        workload.compute_result_with_subgraph(optimized_subgraph, verbose)

        # history.extend(workload)

    @staticmethod
    def extend(history, workload):
        history.extend(workload)

    def cross_optimize(self, history, workload, workload_subgraph, verbose):
        """
        TODO Actual implementation will follow
        also set the data node of the workload graph from the history graph
        :param workload_subgraph:
        :param history:
        :param workload:
        :return:
        """
        self.find_materialized_nodes(history, workload, workload_subgraph, verbose)

    def find_materialized_nodes(self, history, workload, workload_subgraph, verbose):
        """
        TODO this is a bit clunky, at some point we should try to do a better design
        TODO so we don't have to mutate the workload graph inside this function

        find nodes in history graph that can the subgraph view can utilize
        and then copy their data to the workload graph
        :param history:
        :param workload:
        :param workload_subgraph:
        :return:
        """
        # find root nodes of the workload subgraph
        roots = [v for v, d in workload_subgraph.in_degree() if d == 0]
        materialized_nodes = {}
        for r in roots:
            # materialized_nodes are the set of nodes that exists in both graphs and are materialized
            # in the history graph
            materialized_nodes.update(self.find_furthest_materialized_nodes(history.graph, workload_subgraph, r))
        if verbose == 1:
            print 'materialized nodes {}'.format(materialized_nodes)
        for m_w, m_h in materialized_nodes.iteritems():
            history_node = history.graph.nodes[m_h]
            workload_node = workload.graph.nodes[m_w]
            self.copy_from_history(history_node, workload_node)

    @staticmethod
    def find_furthest_materialized_nodes(history_graph, workload_graph, root_vertex):
        """
        given a root vertex and a source graph, find the set of materialized nodes in the target_graph
        that are furthest from the source and exist in both the source and target graph
        the root in the graph and returns vertex ids
        :param history_graph: history graph
        :param workload_graph: workload sub graph
        :param root_vertex:
        :return:
        """
        valid_nodes = [(root_vertex, root_vertex)]
        materialized_nodes = {root_vertex: root_vertex}

        out_degree = {v: d for v, d in workload_graph.out_degree() if d > 0}
        combined_nodes = {}
        potential_valid_super_nodes = {}
        while valid_nodes:
            while valid_nodes:
                w, h = valid_nodes.pop()
                # w_e is a tuple of (source, destination, data_dictionary)
                for _, w_destination, w_edge_data in workload_graph.out_edges(w, data=True):
                    # if the edge is a combine
                    if w_edge_data['name'] == COMBINE_OPERATION_IDENTIFIER:
                        if w_destination in combined_nodes:
                            combined_nodes[w_destination].add((w, h))
                        else:
                            combined_nodes[w_destination] = {(w, h)}
                        if {wn for wn, hn in combined_nodes[w_destination]} == set(
                                workload_graph.node[w_destination]['involved_nodes']):
                            potential_valid_super_nodes[w_destination] = copy.deepcopy(combined_nodes[w_destination])
                    else:
                        for _, h_destination, h_edge_data in history_graph.out_edges(h, data=True):
                            if w_edge_data['hash'] == h_edge_data['hash']:
                                print 'adding ({},{}) to valid nodes'.format(w_destination, h_destination)
                                valid_nodes.append((w_destination, h_destination))
                                if history_graph.nodes[h_destination]['data'].computed:
                                    materialized_nodes[w_destination] = h_destination
                                    out_degree[w] -= 1

                if w in out_degree and out_degree[w] == 0:
                    del materialized_nodes[w]
            for super_node, involved in potential_valid_super_nodes.iteritems():
                h_super_node = None
                for w, h in involved:
                    add_to_valid = False
                    w_hash = workload_graph[w][super_node]['hash']
                    for _, h_destination, h_hash in history_graph.out_edges(h, data='hash'):
                        if h_hash == w_hash:
                            if h_super_node is not None:
                                assert h_super_node == h_destination
                            else:
                                h_super_node = h_destination
                            add_to_valid = True
                    if not add_to_valid:
                        break

                h_node = list(history_graph.successors(h_super_node))
                w_node = list(workload_graph.successors(super_node))
                assert len(h_node) == 1
                assert len(w_node) == 1
                h_node = h_node[0]
                w_node = w_node[0]
                valid_nodes.append((w_node, h_node))
                if history_graph.nodes[h_node]['data'].computed:
                    materialized_nodes[w_node] = h_node
                    for w, _ in involved:
                        out_degree[w] -= 1
                        if out_degree[w] == 0:
                            del materialized_nodes[w]
            potential_valid_super_nodes = {}

        return materialized_nodes

    @staticmethod
    def copy_from_history(history_node, workload_node):
        workload_node['data'].computed = True
        workload_node['size'] = history_node['size']
        if history_node['type'] == 'Dataset' or history_node['type'] == 'Feature':
            workload_node['data'].c_name, workload_node['data'].c_hash = copy.deepcopy((
                history_node['data'].c_name, history_node['data'].c_hash))
        else:
            workload_node['data'].data_obj = copy.deepcopy(history_node['data'].data_obj)
