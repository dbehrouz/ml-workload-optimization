from abc import abstractmethod
from heuristics import compute_recreation_cost, compute_vertex_potential


class Materializer:
    def __init__(self, experiment_graph, storage_budget, use_r_cost=True, use_potential=True,
                 use_n_pipelines=False, modify_graph=False, verbose=False):
        self.use_rc = use_r_cost
        self.use_pt = use_potential
        self.use_n_pipelines = use_n_pipelines
        self.modify_graph = modify_graph
        self.storage_budget = storage_budget
        self.verbose = verbose
        self.experiment_graph = experiment_graph
        compute_recreation_cost(self.experiment_graph.graph, modify_graph=True)
        compute_vertex_potential(self.experiment_graph.graph, modify_graph=True)

    def compute_rhos(self):
        rhos = []
        for node in self.experiment_graph.graph.nodes(data=True):
            if node[1]['root']:
                rho = float('inf')

            elif node[1]['type'] == 'SuperNode':
                rho = 0
            elif node[1]['type'] == 'GroupBy':
                # we are not materializing group by nodes at any cost
                rho = 0
                node[1]['data'].clear_content()
                # node[1]['size'] = 0.0
            else:
                if node[1]['size'] == 0.0:
                    print 'nothing to materialize for node {}'.format(node[0])
                    rho = -1
                else:
                    rho_object = RHO(node[0], node[1]['recreation_cost'],
                                     node[1]['potential'],
                                     node[1]['meta_freq'],
                                     node[1]['size'], self.use_rc, self.use_pt)
                    rhos.append(rho_object)
                    rho = rho_object.rho

            if self.modify_graph:
                node[1]['rho'] = rho

        rhos.sort(reverse=True)
        return rhos

    @staticmethod
    def select_nodes_to_materialize(rhos, remaining_budget, materialized_so_far):
        to_mat = list(materialized_so_far)
        current_budget = remaining_budget
        remaining = []

        while len(rhos) > 0:
            top = rhos.pop(0)
            if top.node_id in to_mat:
                continue
            if current_budget - top.size >= 0:
                to_mat.append(top.node_id)
                current_budget -= top.size
            else:
                remaining.append(top)

        return to_mat, remaining

    def materialize(self, should_materialize):
        """
        materializes the list of vertices in the @should_materialize
        After calling this method, the data manager and history graph of the execution environment will be modified
        :param should_materialize: list of vertices to materialize (vertex ids)
        """
        to_keep = set()
        graph = self.experiment_graph.graph
        data_storage = self.experiment_graph.data_storage
        for node in graph.nodes(data=True):
            if node[1]['type'] == 'Dataset':
                node_data = node[1]['data']
                if node[0] in should_materialize:
                    to_keep = to_keep.union(set(node_data.get_column_hash()))
                    node[1]['mat'] = True
                else:
                    node_data.clear_content()
                    node[1]['mat'] = False
            elif node[1]['type'] == 'Feature':
                node_data = node[1]['data']
                if node[0] in should_materialize:
                    to_keep.add(node_data.get_column_hash())
                    node[1]['mat'] = True
                else:
                    node_data.clear_content()
                    node[1]['mat'] = False

            elif node[0] not in should_materialize:
                node[1]['data'].clear_content()
                node[1]['mat'] = False
                # node[1]['size'] = 0.0
            else:
                node[1]['mat'] = True

        to_delete = []
        for k in data_storage.DATA:
            if k not in to_keep and not k.endswith('_size'):
                to_delete.append(k)

        for k in to_delete:
            del data_storage.DATA[k]
            del data_storage.DATA[k + '_size']

    @abstractmethod
    def run(self):
        """
        the code for finding the nodes to materialize
        this function does not and should not perform the actual materialization
        it only returns the list of the nodes that should be materialized
        :return: a list of (vertex_ids) to be materialized
        """
        pass

    def run_and_materialize(self):
        """
        calls run and then the materialize function
        it 'materializes' the nodes specified by the run method. After calling this method,
        the data manager and history graph of the execution environment will be modified
        """
        should_materialize = self.run()
        self.materialize(should_materialize)


class AllMaterializer(Materializer):
    

class HeuristicsMaterializer(Materializer):

    def run(self):
        """
        simple heuristics based materialization.
        First, it computes all the rhos (feasibility ratio) based on the formula in the paper
        Then, starting fro the highest rhos it starts to materialize until the storage budget is met
        :return:
        """
        if self.use_n_pipelines:
            raise Exception('num pipeline is not supported yet')
        graph = self.experiment_graph.graph
        rhos = self.compute_rhos()
        root_size = 0.0
        to_mat = []
        for n in graph.nodes(data=True):
            if n[1]['root']:
                root_size += n[1]['size']
                to_mat.append(n[0])
        should_materialize, remaining = self.select_nodes_to_materialize(rhos, self.storage_budget - root_size, to_mat)

        total_node_size = self.experiment_graph.get_size_of(should_materialize)
        remaining_budget = self.storage_budget - total_node_size
        if self.verbose:
            print 'state after heuristics based materialization'
            print 'total size of materialized nodes: {}'.format(total_node_size)
            print 'remaining budget: {}, number of nodes to materialize: {}'.format(remaining_budget,
                                                                                    len(should_materialize))

        return should_materialize, remaining


class StorageAwareMaterializer(Materializer):

    def run(self):
        if self.use_n_pipelines:
            raise Exception('num pipeline is not supported yet')

        rhos = self.compute_rhos()
        root_size = 0.0
        to_materialize = []

        for n in self.experiment_graph.graph.nodes(data=True):
            if n[1]['root']:
                root_size += n[1]['size']
                to_materialize.append(n[0])

        remaining_budget = self.storage_budget - root_size
        i = 1
        while remaining_budget >= 0:
            start_list = list(to_materialize)

            to_materialize, rhos = self.select_nodes_to_materialize(rhos, remaining_budget, to_materialize)
            print 'current size: {}'.format(self.experiment_graph.get_size_of(to_materialize))
            # if node new node is materialized, end the process
            if start_list == to_materialize:
                break
            actual_size = self.mock_materialize(to_materialize, rhos)
            remaining_budget = self.storage_budget - actual_size
            if self.verbose:
                total_node_size = self.experiment_graph.get_size_of(to_materialize)
                print 'state after iteration {}'.format(i)
                print 'total size of materialized nodes: {}, actual on disk size: {}'.format(total_node_size,
                                                                                             actual_size)
                print 'remaining budget: {}, number of nodes to materialize: {}'.format(remaining_budget,
                                                                                        len(to_materialize))

            i += 1
        return to_materialize

    def mock_materialize(self, should_materialize, remaining_rhos):
        to_keep = set()
        graph_size = 0

        for node in self.experiment_graph.graph.nodes(data=True):
            if node[1]['type'] == 'Dataset':

                if node[0] in should_materialize:
                    to_keep = to_keep.union(set(node[1]['data'].c_hash))
            elif node[1]['type'] == 'Feature':

                if node[0] in should_materialize:
                    if node[1]['data'].c_hash == '':
                        print 'problem at node {}'.format(node[0])
                    to_keep.add(node[1]['data'].c_hash)
            else:
                if node[0] in should_materialize:
                    graph_size += node[1]['size']

        for rho in remaining_rhos:
            node = self.experiment_graph.graph.nodes[rho.node_id]

            if node['type'] == 'Feature':
                if node['data'].c_hash in to_keep:
                    rho.size = 0.0
            if node['type'] == 'Dataset':
                cols = []
                for c in node['data'].c_hash:
                    if c not in to_keep:
                        cols.append(c)

                rho.size = self.experiment_graph.total_size(column_list=cols)

        return self.experiment_graph.total_size(column_list=to_keep) + graph_size


class RHO(object):
    def __init__(self, node_id, recreation_cost, potential, freq, size, use_rc=True, use_pt=True):
        self.node_id = node_id
        self.recreation_cost = recreation_cost
        self.potential = potential
        self.freq = freq
        self.size = size
        self.rho = self.compute_rho(use_rc, use_pt)

    def compute_rho(self, use_rc, use_pt):
        rho = self.freq
        if use_rc:
            rho *= self.recreation_cost
        if use_pt:
            rho *= self.potential
        rho /= self.size
        return rho

    def __lt__(self, other):
        return self.rho < other.rho

    def __repr__(self):
        return 'node: {}, recreation_cost: {:.3f}, potential: {:.3f}, frequency: {}, size: {:.3f}, rho: {:.5f}' \
            .format(self.node_id[0:12],  # 32 is too long to show
                    self.recreation_cost,
                    self.potential,
                    self.freq,
                    self.size,
                    self.rho)
