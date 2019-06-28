def materialization_simple(ee, storage_budget, use_r_cost=True, use_potential=True, use_n_pipelines=False,
                           modify_graph=True):
    """
    modifies the graph and only keep the data for the
    :type ee: ExecutionEnvironment
    :type modify_graph: bool boolean flag indicating if the graph should be modified with rho and materialization flag
    :type use_potential: bool boolean flag for using potential or not
    :type storage_budget: float storage budget (in MB)
    :type use_n_pipelines: bool bool boolean flag for using number of pipelines a node belongs to or not
    :type use_r_cost: bool boolean flag for using recreation cost or not

    """
    if use_n_pipelines:
        raise Exception('num pipeline is not supported yet')
    graph = ee.history_graph.graph
    rhos = compute_rhos(graph, use_r_cost, use_potential, use_n_pipelines)

    root_size = 0.0
    to_mat = []
    for n in graph.nodes(data=True):
        if n[1]['root']:
            root_size += n[1]['size']
            to_mat.append(n[0])
    return select_nodes_to_materialize(rhos, storage_budget - root_size, to_mat)


def select_nodes_to_materialize(rhos, total_budget, current_materialized):
    to_mat = list(current_materialized)
    current_budget = total_budget
    remaining = []
    while len(rhos) > 0:
        top = rhos.pop()
        if top[0] in to_mat:
            continue
        if current_budget - top[4] >= 0:
            to_mat.append(top[0])
            current_budget -= top[4]
        else:
            remaining.append(top)
    return to_mat, remaining


def compute_rhos(graph, use_r_cost=True, use_potential=True, modify_graph=False):
    rhos = []
    for node in graph.nodes(data=True):
        if node[1]['root']:
            rho = float('inf')

        elif node[1]['type'] == 'SuperNode':
            rho = 0

        else:
            rho = 1.0
            rc = node[1]['recreation_cost']
            pt = node[1]['potential']
            numpip = node[1]['num_pipelines']

            if use_r_cost:
                rho *= rc
            if use_potential:
                rho *= pt
            si = node[1]['size']
            rho /= si

            rhos.append((node[0], rc, pt, numpip, si, rho))
        if modify_graph:
            node[1]['rho'] = rho

    sorted_rhos = sorted(rhos, key=lambda t: t[5], reverse=True)
    return sorted_rhos


# TODO double check, we made some modification
# After every round of materialization, we go through all the computed rhos and modify the sizes of the remaining items
# to how much storage they need to store now, given that their columns in other data frames may have already been stored
def materialization_storage_aware(ee, storage_budget, use_r_cost=True, use_potential=True, use_n_pipelines=False,
                                  modify_graph=True):
    if use_n_pipelines:
        raise Exception('num pipeline is not supported yet')

    rhos = compute_rhos(ee.history_graph.graph, use_r_cost, use_potential, use_n_pipelines)
    root_size = 0.0
    to_materialize = []

    for n in ee.history_graph.graph.nodes(data=True):
        if n[1]['root']:
            root_size += n[1]['size']

            to_materialize.append(n[0])

    current_budget = storage_budget
    i = 1
    while current_budget >= 0:
        print 'iter {}'.format(i)
        start_list = list(to_materialize)
        to_materialize, rhos = select_nodes_to_materialize(rhos, current_budget, to_materialize)
        actual_size, rhos = mock_materialize(ee, to_materialize, rhos)
        current_budget = storage_budget - actual_size
        if start_list == to_materialize:
            break
        i += 1
    return to_materialize


def mock_materialize(ee, should_materialize, remaining_rhos):
    to_keep = set()
    graph_size = 0
    for node in ee.history_graph.graph.nodes(data=True):
        if node[1]['type'] == 'Dataset':

            if node[0] in should_materialize:
                to_keep = to_keep.union(set(node[1]['data'].c_hash))
        elif node[1]['type'] == 'Feature':

            if node[0] in should_materialize:
                to_keep.add(node[1]['data'].c_hash)
        else:
            if node[0] in should_materialize:
                graph_size += node[1]['size']

    for rho in remaining_rhos:
        node = ee.history_graph.graph.nodes[rho[0]]

        if node['type'] == 'Feature':
            if node['data'].c_hash in to_keep:
                rho[4] = 0.0
        if node['type'] == 'Dataset':
            cols = []
            for c in node['data'].c_hash:
                if c not in to_keep:
                    cols.append(c)

            rho[4] = ee.data_storage.total_size(column_list=cols)

    return ee.data_storage.total_size(column_list=to_keep) + graph_size, remaining_rhos


def materialize(ee, should_materialize):
    """

    :type ee: ExecutionEnvironment
    :type should_materialize: object
    """
    to_keep = set()
    graph = ee.history_graph.graph
    data_storage = ee.data_storage
    for node in graph.nodes(data=True):
        if node[1]['type'] == 'Dataset':
            node_data = node[1]['data']
            if node[0] in should_materialize:
                to_keep = to_keep.union(set(node_data.c_hash))
            else:
                node_data.clear_content()
                node[1]['size'] = 0.0
        elif node[1]['type'] == 'Feature':
            node_data = node[1]['data']
            if node[0] in should_materialize:
                to_keep.add(node_data.c_hash)
            else:
                node_data.clear_content()
                node[1]['size'] = 0.0

        elif node[0] not in should_materialize:
            node[1]['data'].clear_content()
            node[1]['size'] = 0.0
    to_delete = []
    for k in data_storage.DATA:
        if k not in to_keep and not k.endswith('_size'):
            to_delete.append(k)

    for k in to_delete:
        del data_storage.DATA[k]
        del data_storage.DATA[k + '_size']
