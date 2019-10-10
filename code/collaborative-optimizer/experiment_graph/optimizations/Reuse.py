import copy
from abc import abstractmethod
from collections import deque

import networkx as nx


class Reuse:
    NAME = 'BASE_REUSE'

    def __init__(self):
        self.history_reads = 0

    @staticmethod
    def get_reuse(reuse_type):
        reuse_type = reuse_type.upper()
        if reuse_type == BottomUpReuse.NAME:
            return BottomUpReuse()
        elif reuse_type == FastBottomUpReuse.NAME:
            return FastBottomUpReuse()
        elif reuse_type == LinearTimeReuse.NAME:
            return LinearTimeReuse()
        else:
            raise Exception('Undefined Reuse type: {}'.format(reuse_type))

    @abstractmethod
    def run(self, vertex, workload, history, verbose):
        """
        method for implementing the reuse algorithm.
        It returns a tuple (materialize_vertices, execution_vertices, history_reads):
            materialize_vertices: set of already materialized vertices from history
            execution_vertices: set of vertices that must be in the execution path
            history_reads: number of reads from history graph
        :param history: history graph
        :param workload: current workload graph
        :param vertex: queried vertex
        :param verbose: verbosity level (0 or 1)
        :return:
        """
        pass

    def is_mat(self, history, vertex):
        self.history_reads += 1
        try:
            return history.nodes[vertex]['mat']
        except KeyError:
            return False

    def in_history_and_mat(self, history, vertex):
        """
        checks if a nodes is in history and if it is materialized and returns the following
        0: if the node is not in history
        1: if the node is in history but it is not materialized
        2: if the node is in history and it is materialized
        :param history:
        :param vertex:
        :return:
        """
        self.history_reads += 1
        try:
            node = history.nodes[vertex]
        except KeyError:
            # the node does not exist in history
            return 0
        if node['mat']:
            # the node is materialized
            return 2
        else:
            # the node exists but it is not materialized
            return 1

    def check_for_warmstarting(self, history, workload, all_models):
        warmstarting_candidates = set()
        for m in all_models:
            training_datasets = list(workload.predecessors(m))
            assert len(training_datasets) == 1
            training_dataset = training_datasets[0]
            if self.in_history_and_mat(history, training_dataset):
                workload_training_edge = workload.edges[training_dataset, m]
                if not workload_training_edge['warm_startable']:
                    continue
                if not workload_training_edge['should_warmstart']:
                    continue
                results = set()
                for _, hm, history_training_edge in history.out_edges(training_dataset, data=True):
                    history_node = history.nodes[hm]
                    if history_node['type'] == 'SK_Model':
                        if not history_training_edge['warm_startable']:
                            continue
                        elif history_training_edge['no_random_state_model'] == \
                                workload_training_edge['no_random_state_model']:
                            results.add((history_training_edge['args']['model'], history_node['score']))
                if results:
                    best_model = -1
                    best_score = -1
                    for model, score in results:
                        if score > best_score:
                            best_score = score
                            best_model = model
                    model_to_warmstart = copy.deepcopy(best_model)
                    model_to_warmstart.random_state = workload_training_edge['random_state']
                    warmstarting_candidates.add((training_dataset, m, model_to_warmstart))
        return warmstarting_candidates


class BottomUpReuse(Reuse):
    NAME = 'BOTTOMUP'

    def __init__(self):
        Reuse.__init__(self)

    def run(self, vertex, workload, history, verbose):
        e_subgraph = workload.compute_execution_subgraph(vertex)
        materialized_vertices, execution_vertices, model_candidates = self.reverse_bfs(terminal=vertex,
                                                                                       workload_subgraph=e_subgraph,
                                                                                       history=history.graph)
        warmstarting_candidates = self.check_for_warmstarting(history.graph, e_subgraph, model_candidates)
        return materialized_vertices, execution_vertices, warmstarting_candidates, self.history_reads

    def reverse_bfs(self, terminal, workload_subgraph, history):
        """
        perform a reverse bfs on workload stop searching further if the node exist in history
        the search doesn't continue on parents of a node which exists in the history
        :param terminal:
        :param workload_subgraph:
        :param history:
        :return:
        """

        materialized_set = set()
        model_candidates = set()
        warmstarting_candidates = set()
        execution_set = {terminal}
        if workload_subgraph.nodes[terminal]['data'].computed:
            return materialized_set, execution_set, warmstarting_candidates
        if self.is_mat(history, terminal):
            return {terminal}, execution_set, warmstarting_candidates
        if workload_subgraph.nodes[terminal]['type'] == 'SK_Model':
            model_candidates.add(terminal)

        prevs = workload_subgraph.predecessors

        queue = deque([(terminal, prevs(terminal))])
        while queue:
            current, prev_nodes_list = queue[0]
            try:

                prev_node = next(prev_nodes_list)
                if prev_node not in execution_set:
                    # The next node should always be added to the execution set even if it is materialized in the
                    # history which results as the first node in the execution path
                    execution_set.add(prev_node)
                    workload_node = workload_subgraph.nodes[prev_node]

                    if workload_node['data'].computed:
                        pass
                    elif self.is_mat(history, prev_node):
                        materialized_set.add(prev_node)
                    else:
                        if workload_node['type'] == 'SK_Model':
                            model_candidates.add(prev_node)
                        queue.append((prev_node, prevs(prev_node)))

            except StopIteration:
                queue.popleft()
        return materialized_set, execution_set, model_candidates


class FastBottomUpReuse(Reuse):
    """
    The difference between fast bottom up and normal bottom up, is that we do not compute the execution subgraph as a
    first step
    """
    NAME = 'FAST_BOTTOMUP'

    def __init__(self):
        Reuse.__init__(self)

    def run(self, vertex, workload_dag, experiment_graph, verbose):
        materialized_vertices, execution_vertices, model_candidates = \
            self.inplace_reverse_bfs(terminal=vertex,
                                     workload_subgraph=workload_dag.graph,
                                     history=experiment_graph.graph)
        warmstarting_candidates = self.check_for_warmstarting(experiment_graph.graph, workload_dag.graph,
                                                              model_candidates)
        return materialized_vertices, execution_vertices, warmstarting_candidates

    def inplace_reverse_bfs(self, terminal, workload_subgraph, history):
        """
        perform a reverse bfs on workload stop searching further if the node exist in history
        the search doesn't continue on parents of a node which exists in the history
        :param terminal:
        :param workload_subgraph:
        :param history:
        :return:
        """

        materialized_set = set()
        model_candidates = set()
        execution_set = {terminal}
        if workload_subgraph.nodes[terminal]['data'].computed:
            return materialized_set, execution_set, model_candidates
        if self.is_mat(history, terminal):
            return {terminal}, execution_set, model_candidates
        if workload_subgraph.nodes[terminal]['type'] == 'SK_Model':
            model_candidates.add(terminal)

        prevs = workload_subgraph.predecessors

        queue = deque([(terminal, prevs(terminal))])
        while queue:
            current, prev_nodes_list = queue[0]
            try:
                prev_node = next(prev_nodes_list)
                if prev_node not in execution_set:
                    # The next node should always be added to the execution set even if it is materialized in the
                    # history which results as the first node in the execution path
                    execution_set.add(prev_node)
                    if workload_subgraph.nodes[prev_node]['data'].computed:
                        pass
                    elif self.is_mat(history, prev_node):
                        materialized_set.add(prev_node)
                    else:
                        if workload_subgraph.nodes[prev_node]['type'] == 'SK_Model':
                            model_candidates.add(prev_node)
                        queue.append((prev_node, prevs(prev_node)))

            except StopIteration:
                queue.popleft()
        return materialized_set, execution_set, model_candidates


class LinearTimeReuse(Reuse):
    NAME = 'LINEAR_TIME_REUSE'

    def __init__(self):
        Reuse.__init__(self)

    def run(self, vertex, workload, history, verbose):
        print 'LINEAR TIME REUSE, {}'.format(vertex)
        workload_subgraph = workload.compute_execution_subgraph(vertex)
        materialized_vertices, all_models = self.forward_pass(workload_subgraph=workload_subgraph,
                                                              e_graph=history.graph, verbose=verbose)
        materialized_vertices, execution_vertices, all_models = self.backward_pass(
            terminal=vertex,
            workload_subgraph=workload_subgraph,
            materialized_vertices=materialized_vertices,
            warmstarting_candidates=all_models,
            verbose=verbose)

        warmstarting_candidates = self.check_for_warmstarting(history.graph, workload_subgraph, all_models)
        if verbose == 1:
            print 'materialized_vertices: {}'.format(materialized_vertices)
            print 'warmstarting_candidates: {}'.format(warmstarting_candidates)
        return materialized_vertices, execution_vertices, warmstarting_candidates

    @staticmethod
    def forward_pass(workload_subgraph, e_graph, verbose):
        """
        performs a conditional search from the root nodes of the subgraph
        unlike reverse_conditional_bfs, the workload subgraph must be previously computed and is guaranteed not to
        contain any nodes that has materialized data
        :param verbose:
        :param workload_subgraph:
        :param e_graph:
        :return:
        """
        materialized_vertices = set()
        warmstarting_candidates = set()
        recreation_costs = {node: -1 for node in workload_subgraph.nodes}
        for n in nx.topological_sort(workload_subgraph):
            if not e_graph.has_node(n):
                # for sk models that are not in experiment graph, we add them to warmstarting candidate
                if workload_subgraph[n]['type'] == 'SK_Model':
                    warmstarting_candidates.add(n)
                continue

            if workload_subgraph.nodes[n]['data'].computed:
                recreation_costs[n] = 0
            else:
                node = e_graph.nodes[n]
                p_costs = sum([recreation_costs[source] for source, _ in e_graph.in_edges(n)])
                execution_cost = node['compute_cost'] + p_costs
                if not node['mat']:

                    recreation_costs[n] = execution_cost
                    # for sk models that are in experiment graph but are not materialized, we add them to materialized
                    # candidates to see if we can warmstart them with a model that is materialized
                    # TODO is this valid?
                    if workload_subgraph.nodes[n]['type'] == 'SK_Model':
                        warmstarting_candidates.add(n)
                elif node['load_cost'] < execution_cost:

                    recreation_costs[n] = node['load_cost']
                    materialized_vertices.add(n)
                else:

                    recreation_costs = execution_cost
        if verbose:
            print 'After forward pass mat_set={}, warm_set={}'.format(materialized_vertices, warmstarting_candidates)
        return materialized_vertices, warmstarting_candidates

    @staticmethod
    def backward_pass(terminal, workload_subgraph, materialized_vertices, warmstarting_candidates, verbose):

        execution_set = set()
        prevs = workload_subgraph.predecessors
        final_materialized_vertices = set()
        final_warmstarting_cadidates = set()
        queue = deque([(terminal, prevs(terminal))])
        while queue:
            current, prev_nodes_list = queue.pop()
            execution_set.add(current)
            if current in materialized_vertices:
                final_materialized_vertices.add(current)
            elif not workload_subgraph.nodes[current]['data'].computed:
                if current in warmstarting_candidates:
                    final_warmstarting_cadidates.add(current)

                    prev_node = next(prev_nodes_list)
                    if prev_node not in execution_set:
                        queue.append((prev_node, prevs(prev_node)))

        if verbose:
            print 'After backward pass mat_set={}, warm_set={}'.format(materialized_vertices, warmstarting_candidates)

        return final_materialized_vertices, execution_set, final_warmstarting_cadidates
