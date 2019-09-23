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
        elif reuse_type == TopDownReuse.NAME:
            return TopDownReuse()
        elif reuse_type == HybridReuse.NAME:
            return HybridReuse()
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
        if node['data'].computed:
            # the node is materialized
            return 2
        else:
            # the node exists but it is not materialized
            return 1

    def check_for_warmstarting(self, history, workload, model_candidates):
        warmstarting_candidates = set()
        for m in model_candidates:
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
        warmstarting_candidates = self.check_for_warmstarting(history, e_subgraph, model_candidates)
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
            return materialized_set, execution_set, warmstarting_candidates, self.history_reads
        if self.is_mat(history, terminal):
            return {terminal}, execution_set, warmstarting_candidates, self.history_reads
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
        warmstarting_candidates = self.check_for_warmstarting(experiment_graph, workload_dag.graph, model_candidates)
        return materialized_vertices, execution_vertices, warmstarting_candidates, self.history_reads

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


class TopDownReuse(Reuse):
    NAME = 'TOPDOWN'

    def __init__(self):
        Reuse.__init__(self)

    def run(self, vertex, workload, history, verbose):
        e_subgraph = workload.compute_execution_subgraph(vertex)
        materialized_vertices, execution_vertices, model_candidates = self.forward_bfs(terminal=vertex,
                                                                                       workload_subgraph=e_subgraph,
                                                                                       history=history.graph)
        warmstarting_candidates = self.check_for_warmstarting(history, e_subgraph, model_candidates)
        return materialized_vertices, execution_vertices, warmstarting_candidates, self.history_reads

    def forward_bfs(self, terminal, workload_subgraph, history):
        """
        performs a conditional search from the root nodes of the subgraph
        unlike reverse_conditional_bfs, the workload subgraph must be previously computed and is guaranteed not to
        contain any nodes that has materialized data
        :param terminal:
        :param workload_subgraph:
        :param history:
        :return:
        """
        roots = [n for n, d in workload_subgraph.in_degree() if d == 0]

        def forward_bfs(source, candidates_so_far, models_so_far):
            """
            simple forward bfs starting from a source (root node)
            :param candidates_so_far:
            :param source:
            :return:
            """

            materialized_in_this_path = set()
            model_in_this_path = set()
            status = self.in_history_and_mat(history, source)
            if workload_subgraph.nodes[source]['type'] == 'SK_Model':
                model_in_this_path.add(source)
            if status == 2:
                materialized_in_this_path.add(source)
            elif status == 1:
                pass
            elif status == 0:
                return materialized_in_this_path, model_in_this_path
            else:
                raise Exception('invalid status from history read')

            successor = workload_subgraph.successors
            queue = deque([(source, successor(source))])
            visited = {source}
            while queue:
                current, next_nodes_list = queue[0]
                try:
                    next_node = next(next_nodes_list)
                    if next_node not in model_in_this_path:
                        if workload_subgraph.nodes[next_node]['type'] == 'SK_Model':
                            model_candidates.add(next_node)
                    if next_node not in visited:
                        if next_node not in candidates_so_far:
                            # if next_node is in materialized_candidates, this indicates that we have traversed down
                            # this path when performing bfs for another root and we can stop here to save time
                            status = self.in_history_and_mat(history, next_node)
                            if status == 2:
                                # The nodes is materialized
                                # therefore, first add the node to the list and continue the search
                                materialized_in_this_path.add(next_node)
                                queue.append((next_node, successor(next_node)))
                            elif status == 1:
                                # The node is in history but it is not materialized
                                # do not add the node to the candidates but continue the search
                                queue.append((next_node, successor(next_node)))
                            elif status == 0:
                                # The node is not in history, do not continue down this path
                                pass
                            else:
                                raise Exception('invalid status from history read')
                except StopIteration:
                    queue.popleft()

            return materialized_in_this_path, model_in_this_path

        model_candidates = set()
        materialized_candidates = set()
        # for every root node traverse the graph and find the set of candidates
        for r in roots:
            materialized_vertices, model_vertices = forward_bfs(r, materialized_candidates, model_candidates)
            materialized_candidates.union(materialized_vertices)
            model_candidates.union(model_vertices)

        if not materialized_candidates:
            # no materialized candidates could be found
            # return the nodes of the original workload graph
            return set(), set(workload_subgraph.nodes()), self.history_reads

        # Now that we have all the candidate nodes that are materialized in history graph
        # we can do a reverse bfs to construct the final execution path and the materialized nodees
        # that should be returned to the optimizer
        final_mat_candidates = set()
        final_execution_candidates = {terminal}
        if terminal in materialized_candidates:
            return {terminal}, {terminal}, self.history_reads

        prevs = workload_subgraph.predecessors
        queue = deque([(terminal, prevs(terminal))])
        while queue:
            current, prev_nodes_list = queue[0]
            try:
                prev_node = next(prev_nodes_list)
                if prev_node not in final_execution_candidates:
                    final_execution_candidates.add(prev_node)
                    if prev_node in materialized_candidates:
                        final_mat_candidates.add(prev_node)
                    else:
                        queue.append((prev_node, prevs(prev_node)))

            except StopIteration:
                queue.popleft()

        return final_mat_candidates, final_execution_candidates, model_candidates


class HybridReuse(Reuse):
    NAME = 'HYBRID'

    def __init__(self, graph_length_cutoff=10):
        Reuse.__init__(self)

    def run(self, vertex, workload, history, verbose):
        e_subgraph = workload.compute_execution_subgraph(vertex)
        graph_size = len(e_subgraph)
        step_size = int(graph_size / 2)

        self.bottomup()

    def bottomup(self, terminal, workload, history, step_size):
        prevs = workload.predecessors
        queue = deque([(terminal, prevs(terminal))])
        visited = {terminal}
        i = 0
        new_node = None
        while queue:
            current, prev_nodes_list = queue[0]
            try:
                prev_node = next(prev_nodes_list)
                if prev_node not in visited:
                    if i == step_size:
                        new_node = prev_node
                        break
                    else:
                        queue.append((prev_node, prevs(prev_node)))
                        i += 1

            except StopIteration:
                queue.popleft()
