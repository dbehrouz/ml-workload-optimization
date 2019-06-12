"""
The optimizer module performs the scheduling.
It receives the workload execution graph and the historical execution graph and optimizes
the workload execution graph and returns the scheduled execution path
"""
import copy


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

        :type workload: ExecutionGraph
        :param history:
        :param workload:
        :return:
        """
        workload_subgraph = workload.compute_execution_subgraph(v_id)

        # here we optimize workload graph with historical graph
        if not history.is_empty():
            self.cross_optimize(history, workload, workload_subgraph)

        optimized_subgraph = workload.compute_execution_subgraph(v_id)

        print 'size of the optimized graph = {}'.format(len(optimized_subgraph))

        for n in workload.graph.nodes(data='data'):
            print n[0],n[1].computed
        # execute the workload using the optimized view
        workload.compute_result_with_subgraph(optimized_subgraph, verbose)

        # history.extend(workload)

    @staticmethod
    def extend(history, workload):
        history.extend(workload)

    def cross_optimize(self, history, workload, workload_subgraph):
        """
        TODO Actual implementation will follow
        also set the data node of the workload graph from the history graph
        :param workload_subgraph:
        :param history:
        :param workload:
        :return:
        """
        self.find_materialized_nodes(history, workload, workload_subgraph)

    def find_materialized_nodes(self, history, workload, workload_subgraph):
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
        for r in roots:
            # materialized_nodes are the set of nodes that exists in both graphs and are materialized
            # in the history graph
            materialized_nodes = self.find_furthest_materialized_nodes(history.graph, workload_subgraph, r)
            print 'materialized nodes = {}'.format(materialized_nodes)
            for m_w, m_h in materialized_nodes.iteritems():
                history_node = history.graph.nodes[m_h]
                workload_node = workload.graph.nodes[m_w]
                self.copy_from_history(history_node, workload_node)

    @staticmethod
    def find_furthest_materialized_nodes(target_graph, source_graph, root_vertex):
        """
        given a root vertex and a source graph, find the set of materialized nodes in the target_graph
        that are furthest from the source and exist in both the source and target graph
        the root in the graph and returns vertex ids
        :param target_graph: history graph
        :param source_graph: workload sub graph
        :param root_vertex:
        :return:
        """
        valid_nodes = [(root_vertex, root_vertex)]
        materialized_nodes = {root_vertex: root_vertex}
        new_vertices = []
        # cur_node_history = root_vertex
        # matching_nodes = {}
        i = 1
        while len(valid_nodes) > 0:
            #print 'valid nodes at step {} = {}'.format(i, valid_nodes)
            for w, h in valid_nodes:
                new_vertices = []
                out_edges = source_graph.out_edges(w, data='hash')
                all_edges_exist = True
                for w_e in out_edges:
                    #print 'out edge in workload graph at step {} = {}'.format(i, w_e)
                    for h_e in target_graph.out_edges(h, data='hash'):
                        #print 'out edge in history graph step {} = {}'.format(i, w_e)
                        if w_e[2] == h_e[2]:
                            #print 'same node exists in both graphs'
                            if target_graph.nodes[h_e[1]]['data'].computed:
                                new_vertices.append((w_e[1], h_e[1]))
                                materialized_nodes[w_e[1]] = h_e[1]
                        else:
                            all_edges_exist = False
                #if not all_edges_exist:
                    #del materialized_nodes[w]

            valid_nodes = list(new_vertices)
            i = i + 1

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
