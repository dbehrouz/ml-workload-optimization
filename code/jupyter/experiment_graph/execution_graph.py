import copy
import sys

import networkx as nx
import pandas as pd

# Reserved word for representing super nodes.
# Do not use combine as an operation name
# TODO: make file with all the global names
COMBINE_OPERATION_IDENTIFIER = 'combine'
AS_MB = 1024.0 * 1024.0


class ExecutionGraph(object):
    roots = []

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id, **meta):
        self.graph.add_node(node_id, **meta)

    def add_edge(self, start_id, end_id, nextnode, meta, ntype):
        for e in self.graph.out_edges(start_id, data=True):
            if e[2]['hash'] == meta['hash']:
                exist = self.graph.nodes[e[1]]['data']
                e[2]['freq'] = e[2]['freq'] + 1
                return exist

        self.add_node(end_id, **{'type': ntype, 'root': False, 'data': nextnode, 'size': 0.0})
        meta['freq'] = 1
        self.graph.add_edge(start_id, end_id, **meta)
        return None

    def plot_graph(self, plt):
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph)
        # pos = graphviz_layout(ee.graph.graph, prog='twopi', args='')
        color_map = []
        for node in self.graph.nodes(data=True):

            if node[1]['root']:
                color_map.append('green')
            elif node[1]['type'] == 'Dataset' or node[1]['type'] == 'Feature':
                color_map.append('red')
            elif node[1]['type'] == 'Agg' or node[1]['type'] == 'SK_Model':
                color_map.append('blue')
            elif node[1]['type'] == 'SuperNode':
                color_map.append('grey')
            else:
                color_map.append('black')

        nx.draw(self.graph,
                node_color=color_map,
                pos=pos,
                node_size=100)
        nx.draw_networkx_edge_labels(self.graph,
                                     pos=pos,
                                     edge_labels={(u, v): d["name"] for u, v, d in self.graph.edges(data=True)})

    @staticmethod
    def compute_size(data):
        if isinstance(data, pd.DataFrame):
            return sum(data.memory_usage(index=True, deep=True)) / AS_MB
        elif isinstance(data, pd.Series):
            return data.memory_usage(index=True, deep=True) / AS_MB
        else:
            return sys.getsizeof(data) / AS_MB

    def get_total_size(self):
        t_size = 0
        for node in self.graph.nodes(data=True):
            t_size += node[1]['size']
            # t_size += self.compute_size(node[1]['data'].data)
        return t_size

    def has_node(self, node_id):
        return self.graph.has_node(node_id)

    def brute_force_compute_paths(self, vertex):
        """brute force method for computing all the paths

        :param vertex: the vertex that should be materialized
        :return: path in the form of [(i,j)] indicating the list of edges that should be executed
        """

        def tuple_list(li):
            res = []
            for i in range(len(li) - 1):
                res.append((li[i], li[i + 1]))
            return res

        all_simple_paths = []
        for s in self.roots:
            for path in nx.all_simple_paths(self.graph, source=s, target=vertex):
                all_simple_paths.append(path)
        # for every path find the sub path that is not computed yet
        all_paths = []
        for path in all_simple_paths:
            cur_index = len(path) - 1
            while self.graph.nodes[path[cur_index]]['data'].is_empty():
                cur_index -= 1
            all_paths.append(path[cur_index:])

        tuple_set = []
        for path in all_paths:
            tuple_set.append(tuple_list(path))
        flatten = [item for t in tuple_set for item in t]

        return flatten

    def fast_compute_paths(self, vertex):
        """faster alternative to brute_force_compute_paths
        instead of finding all the path in the graph, in this method, we traverse backward from the destination node
        so computing the path of the nodes that are not already materialized.

        :param vertex: the vertex that should be materialized
        :return: path in the form of [(i,j)] indicating the list of edges that should be executed
        """

        def get_path(source, paths):
            for v in self.graph.predecessors(source):
                paths.append((v, source))
                if self.graph.nodes[v]['data'].is_empty():
                    get_path(v, paths)

        all_paths = []
        get_path(vertex, all_paths)
        return all_paths

    def compute_result(self, v_id, verbose=0):
        """ main computation for nodes
            This functions uses the schedule provided by the scheduler functions
            (currently: fast_compute_paths, brute_force_compute_paths) to compute
            the requested node
        """
        # compute_paths = self.brute_force_compute_paths(v_id)
        compute_paths = self.fast_compute_paths(v_id)

        # schedule the computation of nodes
        schedule = self.schedule(compute_paths)

        # execute the computation based on the schedule
        cur_node = None
        for pair in schedule:
            cur_node = self.graph.nodes[pair[1]]
            edge = self.graph.edges[pair[0], pair[1]]
            # print the path while executing
            if verbose == 1:
                print str(pair[0]) + '--' + edge['hash'] + '->' + str(pair[1])
                # combine is logical and we do not execute it
            if edge['oper'] != COMBINE_OPERATION_IDENTIFIER:
                # Assignment wont work since it copies object reference
                # TODO: check if a shallow copy is enough
                cur_node['data'].data = copy.deepcopy(self.compute_next(self.graph.nodes[pair[0]], edge))
                cur_node['size'] = self.compute_size(cur_node['data'].data)

        return cur_node['data']

    @staticmethod
    def compute_next(node, edge):
        func = getattr(node['data'], edge['oper'])
        return func(**edge['args'])

    @staticmethod
    def schedule(path):
        """schedule the computation of nodes
        receives all the paths that should be computed. Every path starts with
        a node that is already computed.
        It returns a list of tuples which specifies the execution order of the nodes
        the list is of the form [(i,j), ...], where node[j] = node[i].operation, where operation
        is specifies inside the edge (i,j)
        :param path: a list of edges (i,j) which indicates the operations that should be executed
        :return: the ordered list of edges to executed
        """

        def get_end_point(endnode, li):
            for i in range(len(li)):
                if endnode == li[i][1]:
                    return li[i]
            return -1

        def is_feasible(li):
            """Check if a schedule is feasible
            A schedule is feasible if for every start node at position i
            there is no end node at positions greater than i.
            This essentially means, if a node is the source of a computation, it must be 
            computed beforehand
            """
            for i in range(len(li)):
                for j in range(i, len(li)):
                    if get_end_point(li[i][0], li[j:]) != -1:
                        return False
            return True

        schedule = []
        # removing overlapping edges resulted from multiple paths from the root to the end node
        for i in path:
            if i not in schedule:
                schedule.append(i)
        # TODO: this can be done more efficiently
        while not is_feasible(schedule):
            for i in range(len(schedule)):
                toswap = get_end_point(schedule[i][0], schedule[i:])
                if toswap != -1:
                    schedule.remove(toswap)
                    schedule.insert(i, toswap)
        return schedule
