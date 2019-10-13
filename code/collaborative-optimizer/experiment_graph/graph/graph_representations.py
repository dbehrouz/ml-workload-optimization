import copy
from collections import deque
from datetime import datetime

import networkx as nx
import numpy as np

# Reserved word for representing super graph.
# Do not use combine as an operation name
# TODO: make file with all the global names
from auxilary import DataFrame, DataSeries
from experiment_graph.data_storage import DedupedStorageManager, SimpleStorageManager
from experiment_graph.globals import COMBINE_OPERATION_IDENTIFIER
from node import Feature, Dataset


class BaseGraph(object):
    def __init__(self, graph, roots):
        if graph is None:
            self.graph = nx.DiGraph()
        else:
            self.graph = graph
        if roots is None:
            self.roots = []
        else:
            self.roots = roots

    def set_environment(self, env):
        for node in self.graph.nodes(data='data'):
            print node
            node[1].execution_environment = env

    def is_empty(self):
        return len(self.graph) == 0

    def add_node(self, node_id, **meta):
        self.graph.add_node(node_id, **meta)

    def add_edge(self, start_id, end_id, nextnode, meta, ntype):
        for e in self.graph.out_edges(start_id, data=True):
            if e[2]['hash'] == meta['hash']:
                exist = self.graph.nodes[e[1]]['data']
                e[2]['freq'] = e[2]['freq'] + 1
                return exist
        params = {'type': ntype, 'root': False, 'data': nextnode, 'size': 0.0}

        if ntype == 'SK_Model':
            params['score'] = -1

        self.add_node(end_id, **params)
        meta['freq'] = 1
        self.graph.add_edge(start_id, end_id, **meta)
        return None

    def plot_graph(self, plt, figsize=(12, 12), labels_for_vertex=['size'], labels_for_edges=['name'], vertex_size=1000,
                   vertex_font_size=10):
        """
        plot the graph using the graphvix dot layout
        :param vertex_font_size:
        :param vertex_size:
        :param labels_for_edges:
        :param labels_for_vertex:
        :param figsize: size of the figure (default (12,12))
        :param plt: matlibplot object
        """
        from networkx.drawing.nx_agraph import graphviz_layout
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(1, 1, 1)
        pos = graphviz_layout(self.graph, prog='dot', args='')

        # get the list of available types and frequency of each node
        vertex_labels = {}
        unique_types = []

        for node in self.graph.nodes(data=True):
            if node[1]['type'] not in unique_types:
                unique_types.append(node[1]['type'])
            labels = []
            for p in labels_for_vertex:

                if p == 'id':
                    if node[1]['root']:
                        labels.append('root')
                    else:
                        labels.append(node[0][:10])
                if p not in node[1]:
                    labels.append('-')
                elif type(node[1][p]) is np.float64 or type(node[1][p]) is float:
                    labels.append('{:.3f}'.format(node[1][p]))
                else:
                    labels.append(str(node[1][p]))

            vertex_labels[node[0]] = ','.join(labels)

        jet = plt.get_cmap('gist_rainbow')
        colors = jet(np.linspace(0, 1, len(unique_types)))
        color_map = dict(zip(unique_types, colors))
        for label in color_map:
            ax.scatter(None, None, color=color_map[label], label=label)

        # TODO there's a problem with nodelist=...., the node type and legends dont match
        materialized_nodes = [n[0] for n in self.graph.node(data='mat') if n[1]]
        all_colors = [color_map[n[1]['type']] for n in self.graph.nodes(data=True) if n[1]['mat']]
        nx.draw_networkx(
            self.graph,
            node_size=vertex_size,
            nodelist=materialized_nodes,
            cmap=jet,
            # vmin=0,
            # vmax=len(unique_types),
            node_color=all_colors,
            node_shape='s',
            pos=pos,
            with_labels=False,
            ax=ax)

        non_materialized_nodes = [n[0] for n in self.graph.node(data='mat') if not n[1]]
        all_colors = [color_map[n[1]['type']] for n in self.graph.nodes(data=True) if not n[1]['mat']]
        nx.draw_networkx(
            self.graph,
            edgelist=[],
            node_size=vertex_size,
            nodelist=non_materialized_nodes,
            cmap=jet,
            # vmin=0,
            # vmax=len(unique_types),
            node_color=all_colors,
            node_shape='o',
            pos=pos,
            with_labels=False,
            ax=ax)

        if labels_for_vertex:
            nx.draw_networkx_labels(self.graph,
                                    pos=pos,
                                    labels=vertex_labels,
                                    font_size=vertex_font_size)

        def construct_label(edge_data, edge_labels):
            return ','.join(['' if str(edge_data[l]) == 'combine' else str(edge_data[l]) for l in edge_labels])

        nx.draw_networkx_edge_labels(
            self.graph,
            pos=pos,
            edge_labels={(u, v): construct_label(d, labels_for_edges) for u, v, d in self.graph.edges(data=True)})

        plt.axis('off')
        f.set_facecolor('w')
        leg = ax.legend(markerscale=4, loc='best', fontsize=12, scatterpoints=1)

        for line in leg.get_lines():
            line.set_linewidth(4.0)

    def get_artifact_sizes(self, for_types=None, exclude_types=None, mat_only=False):
        def none_to_zero(value):
            return 0 if value is None else value

        t_size = 0
        # both cannot have some values
        assert for_types is None or exclude_types is None

        if exclude_types is None:
            if for_types is None:
                for node in self.graph.nodes(data=True):
                    if not mat_only or node[1]['mat']:
                        t_size += none_to_zero(node[1]['size'])
            else:
                for node in self.graph.nodes(data=True):
                    if node[1]['type'] in for_types:
                        if not mat_only or node[1]['mat']:
                            t_size += none_to_zero(node[1]['size'])
        else:
            for node in self.graph.nodes(data=True):
                if node[1]['type'] not in exclude_types:
                    if not mat_only or node[1]['mat']:
                        t_size += none_to_zero(node[1]['size'])

        return t_size

    def get_total_materialized_size(self):
        # excluding groupby is not needed since we are setting its size to zero
        # but this is just to confirm that groupby is never supposed ot be materialized
        # TODO remove groupby from exclusion when it is really removed from the graph
        return self.get_artifact_sizes(exclude_types='GroupBy', mat_only=True)

    def get_total_size(self):
        return self.get_artifact_sizes(exclude_types='GroupBy')

    def get_size_of(self, node_list):
        size = 0.0
        for n in node_list:
            size += self.graph.nodes[n]['size']
        return size

    def has_node(self, node_id):
        return self.graph.has_node(node_id)

    def get_node(self, node_id):
        return self.graph.nodes[node_id]


class WorkloadDag(BaseGraph):
    def __init__(self, graph=None, roots=None):
        super(WorkloadDag, self).__init__(graph, roots)
        self.post_processed = False

    def post_process(self):
        """
        This method should be call before extending the history to add quality and frequency to the graph
        :return:
        """
        # for node in self.graph.nodes(data=True):
        #     node[1]['freq'] = node[1]['data'].get_freq()
        prev_node = None
        for n in list(nx.topological_sort(self.graph)):
            node = self.graph.nodes[n]
            if node['data'].computed:
                if node['type'] == 'SK_Model':
                    node['score'] = node['data'].get_model_score()
                if node['type'] == 'GroupBy':
                    node['size'] = prev_node['size']
                elif node['type'] is not 'SuperNode':
                    node['size'] = node['data'].compute_size()
                else:
                    node['size'] = None
                prev_node = node
            else:
                node['size'] = None

        self.post_processed = True

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
            while not self.graph.nodes[path[cur_index]]['data'].computed:
                cur_index -= 1
            all_paths.append(path[cur_index:])

        tuple_set = []
        for path in all_paths:
            tuple_set.append(tuple_list(path))
        flatten = [item for t in tuple_set for item in t]

        return flatten

    def compute_execution_subgraph(self, vertex):
        """this method performs a similar job to the brute force and fast compute paths functions. However, instead of
        just returning a list of [(source,destination)] vertex pairs, it returns the actual subgraph
        This way we can also use the subgraph for cross optimizing with the history graph
        :param vertex: vertex we are trying to compute
        :return: subgraph that must be computed
        """

        execution_set = {vertex}
        if not self.graph.nodes[vertex]['data'].computed:
            prevs = self.graph.predecessors

            queue = deque([(vertex, prevs(vertex))])
            while queue:
                current, prev_nodes_list = queue[0]
                try:
                    prev_node = next(prev_nodes_list)
                    if prev_node not in execution_set:
                        execution_set.add(prev_node)
                        if not self.graph.nodes[prev_node]['data'].computed:
                            queue.append((prev_node, prevs(prev_node)))

                except StopIteration:
                    queue.popleft()

        # TODO we should check to make sure the subgraph induction is not slow
        # TODO otherwise we can compute the subgraph directly when finding the vertices
        return self.graph.subgraph(execution_set)

    def fast_compute_paths(self, vertex):
        """faster alternative to brute_force_compute_paths
        instead of finding all the path in the graph, in this method, we traverse backward from the destination node
        so computing the path of the graph that are not already materialized.

        :param vertex: the vertex that should be materialized
        :return: path in the form of [(i,j)] indicating the list of edges that should be executed
        """

        def get_path(source, paths):
            if not self.graph.nodes[source]['data'].computed:
                for v in self.graph.predecessors(source):
                    paths.append((v, source))
                    if not self.graph.nodes[v]['data'].computed:
                        get_path(v, paths)

        all_paths = []
        get_path(vertex, all_paths)
        return all_paths

    def compute_result_with_subgraph(self, subgraph, verbose=0):
        """
        :param subgraph:
        :param verbose:
        :return:
        """
        # schedule the computation of graph
        # schedule = self.schedule(compute_paths)
        schedule = list(nx.topological_sort(nx.line_graph(subgraph)))

        # execute the computation based on the schedule
        for pair in schedule:
            cur_node = self.graph.nodes[pair[1]]
            prev_node = self.graph.nodes[pair[0]]
            edge = self.graph.edges[pair[0], pair[1]]

            # combine is logical and we do not execute it
            if edge['oper'] != COMBINE_OPERATION_IDENTIFIER:
                if not cur_node['data'].computed:
                    # print the path while executing
                    if verbose == 1:
                        print str(pair[0]) + '--' + edge['hash'] + '->' + str(pair[1])
                    # TODO: Data Storage only stores the data for Dataset and Feature for now
                    # TODO: Later on maybe we want to consider storing models and aggregates on the data storage as well
                    if cur_node['type'] == 'Dataset' or cur_node['type'] == 'Feature':
                        start_time = datetime.now()
                        cur_node['data'].underlying_data = self.compute_next(prev_node, edge)
                        cur_node['data'].computed = True
                        total_time = (datetime.now() - start_time).microseconds / 1000.0
                        # cur_node['size'] = cur_node['data'].compute_size()
                    # all the other node types they contain the data themselves
                    else:
                        start_time = datetime.now()
                        cur_node['data'].underlying_data = self.compute_next(prev_node, edge)
                        cur_node['data'].computed = True
                        total_time = (datetime.now() - start_time).microseconds / 1000.0

                    edge['execution_time'] = total_time
                    edge['executed'] = True
            else:
                edge['execution_time'] = 0.0
                edge['executed'] = True
                cur_node['size'] = None
        return schedule

    def compute_result(self, v_id, verbose=0):
        """ main computation for graph
            This functions uses the schedule provided by the scheduler functions
            (currently: fast_compute_paths, brute_force_compute_paths) to compute
            the requested node
        """
        # compute_paths = self.brute_force_compute_paths(v_id)
        # compute_paths = self.fast_compute_paths(v_id)
        execution_subgraph = self.compute_execution_subgraph(v_id)

        self.compute_result_with_subgraph(execution_subgraph, verbose)

    @staticmethod
    def compute_next(node, edge):
        func = getattr(node['data'], edge['oper'])
        return func(**edge['args'])

    @staticmethod
    def schedule(path):
        """schedule the computation of graph
        receives all the paths that should be computed. Every path starts with
        a node that is already computed.
        It returns a list of tuples which specifies the execution order of the graph
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


class ExperimentGraph(BaseGraph):
    def __init__(self, data_storage=SimpleStorageManager(), graph=None, roots=None):
        super(ExperimentGraph, self).__init__(graph, roots)
        self.data_storage = data_storage

    def retrieve_data(self, node_id):
        node = self.graph.nodes[node_id]
        if node['mat'] is not True:
            raise Exception('The node ({}) is not materialized'.format(node_id))

        if node['type'] == 'Dataset':
            return DataFrame(column_names=node['data'].underlying_data.get_column(),
                             column_hashes=node['data'].underlying_data.get_column_hash(),
                             pandas_df=self.data_storage.get(node_id))
        elif node['type'] == 'Feature':
            return DataSeries(column_name=node['data'].underlying_data.get_column(),
                              column_hash=node['data'].underlying_data.get_column_hash(),
                              pandas_series=self.data_storage.get(node_id))
        else:
            return copy.deepcopy(self.graph.nodes[node_id]['data'].underlying_data)

    def get_real_size(self):
        return self.get_artifact_sizes(exclude_types=['Dataset', 'Feature'],
                                       mat_only=True) + self.data_storage.total_size()

    def materialize(self, node_id, artifact):
        if node_id not in self.graph:
            raise Exception('Node not present in the Experiment Graph')
        else:
            if not artifact.computed:
                print 'warning: artifact {} is never computed in the workload dag'.format(node_id)
                return
            node = self.graph.nodes[node_id]
            if node['type'] == 'Dataset':
                assert isinstance(artifact, Dataset)
                # or node['type'] == 'Feature':
                self.data_storage.put(node_id, artifact.underlying_data)
                # artifact.underlying_data.pandas_df = None
                node['data'] = copy.copy(artifact)

            elif node['type'] == 'Feature':
                assert isinstance(artifact, Feature)
                self.data_storage.put(node_id, artifact.underlying_data)
                # artifact.underlying_data.pandas_series = None
                node['data'] = copy.copy(artifact)
            else:
                node['data'] = copy.copy(artifact)

            node['mat'] = True

    def unmaterialize(self, node_id):
        if node_id not in self.graph:
            raise Exception('Node not present in the Experiment Graph')
        else:
            node = self.graph.nodes[node_id]
            if node['type'] == 'Dataset' or node['type'] == 'Feature':
                self.data_storage.delete(node_id)
            else:
                node['data'].remove_content()
            node['mat'] = False

    def extend(self, workload):
        # make sure the workload graph is post processed, i.e., the model scores are added to the graph
        if not workload.post_processed:
            raise Exception('Workload is not post processed')

        self.roots = list(set(self.roots + workload.roots))

        for node_id, node_attributes in workload.graph.nodes(data=True):
            # Only artifacts which are computed should be updated
            if node_attributes['data'].computed:
                self.add_node_to_experiment_graph(node_id, node_attributes)
                self.compute_load_cost(node_id, node_attributes['data'])

            # Supernodes by are never computed, we add them when all the incoming nodes should be computed
            elif node_attributes['type'] == 'SuperNode':
                involved_nodes = node_attributes['involved_nodes']
                should_add = True
                for n in involved_nodes:
                    if not workload.graph.nodes[n]['data'].computed:
                        should_add = False
                if should_add:
                    self.add_node_to_experiment_graph(node_id, node_attributes)

        for s, d, data in workload.graph.edges(data=True):
            # TODO make a proper edge and node classes
            # print s,d,data
            if data['executed']:
                if s in self.graph.nodes and d in self.graph.nodes:
                    self.graph.add_edge(s, d, **data)
        import gc
        gc.collect()

        # self.graph.add_edges_from(workload.graph.edges(data=True))

    def add_node_to_experiment_graph(self, node_id, node_attributes):
        if node_id not in self.graph.nodes:
            eg_attributes = {}
            for k, v in node_attributes.iteritems():
                if k != 'data':
                    eg_attributes[k] = copy.deepcopy(v)
            eg_attributes['data'] = None
            eg_attributes['meta_freq'] = 1
            eg_attributes['mat'] = False
            self.graph.add_node(node_id, **eg_attributes)
        else:
            self.graph.nodes[node_id]['meta_freq'] += 1

    def compute_load_cost(self, node_id, artifact):
        """
        Since using a profile to estimate the load cost gives us not accurate result, we actually add a node
        compute its load cost and then remove it from the graph
        :param workload:
        :return:
        """

        if not self.graph.has_node(node_id):
            raise Exception('Every vertex should be in Experiment Graph by now !!!')

        if 'load_cost' not in self.graph.nodes[node_id]:
            self.materialize(node_id, artifact)
            start = datetime.now()
            dummy = self.retrieve_data(node_id)
            load_time = ((datetime.now() - start).microseconds / 1000.0)
            self.unmaterialize(node_id)
            del dummy
            self.graph.nodes[node_id]['load_cost'] = load_time
