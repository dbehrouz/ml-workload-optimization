import networkx as nx


class ExecutionGraph(object):
    roots = []

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, id, **meta):
        self.graph.add_node(id, **meta)

    def add_edge(self, start_id, end_id, nextnode, meta, ntype):
        exist = None
        for e in self.graph.out_edges(start_id, data=True):
            if e[2]['hash'] == meta['hash']:
                exist = self.graph.nodes[e[1]]['data']
                e[2]['freq'] = e[2]['freq'] + 1
                return exist

        self.add_node(end_id, **{'type': ntype, 'root': False, 'data': nextnode})
        meta['freq'] = 1
        self.graph.add_edge(start_id, end_id, **meta)
        return None

    def has_node(self, node_id):
        return self.graph.has_node(node_id)

    def compute_result(self, v_id):
        """ main computation for nodes
                
        """
        # find every path from root to the current node
        p = []
        for s in self.roots:
            for path in nx.all_simple_paths(self.graph, source=s, target=v_id):
                p.append(path)

        # for every path find the sub path that is not computed yet
        compute_paths = []
        for path in p:
            cur_index = len(path) - 1
            while self.graph.nodes[path[cur_index]]['data'].is_empty():
                cur_index -= 1
            compute_paths.append(path[cur_index:])

        # schedule the computation of nodes
        schedule = self.schedule(compute_paths)

        # execute the computation based on the schedule
        cur_node = None
        for pair in schedule:
            cur_node = self.graph.nodes[pair[1]]
            edge = self.graph.edges[pair[0], pair[1]]
            # merge is logical and we do not execute it 
            if edge['oper'] != 'merge':
                cur_node['data'].data = self.compute_next(self.graph.nodes[pair[0]], edge)

        return cur_node['data']

    def compute_next(self, node, edge):
        func = getattr(node['data'], edge['oper'])
        return func(**edge['args'])

    def schedule(self, paths):
        """ schedule the computationg of nodes
        receives all the paths that should be computed. Every path starts with
        a node that is already computed.
        It returns a list of tuples which specifies the execution order of the nodes
        the list is of the form [(i,j), ...], where node[j] = node[i].operation, where operation
        is specifies inside the edge (i,j)
        """

        def tuple_list(li):
            res = []
            for i in range(len(li) - 1):
                res.append((li[i], li[i + 1]))
            return res

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
            computed befrehand
            """
            for i in range(len(li)):
                for j in range(i, len(li)):
                    if get_end_point(li[i][0], li[j:]) != -1:
                        return False
            return True

        schedule = []
        if len(paths) == 1:
            path = paths[0]
            for i in range(len(path) - 1):
                schedule.append((path[i], path[i + 1]))
        else:
            # transform to tuples (source, destination)        
            tuple_set = []
            for path in paths:
                tuple_set.append(tuple_list(path))
            flatten = [item for t in tuple_set for item in t]
            # Make sure no operation is repeated
            # after this line, tuple repeation is not allowed
            for i in flatten:
                if i not in schedule:
                    schedule.append(i)

            while not is_feasible(schedule):
                for i in range(len(schedule)):
                    toswap = get_end_point(schedule[i][0], schedule[i:])
                    if toswap != -1:
                        schedule.remove(toswap)
                        schedule.insert(i, toswap)
        return schedule
