import networkx as nx


def compute_load_costs(graph, cost_profile):
    for n, d in graph.nodes(data=True):
        if 'load_cost' not in d:
            if d['type'] != 'SuperNode' and d['type'] != 'GroupBy':
                d['load_cost'] = d['size'] * cost_profile[d['type']]


def compute_recreation_cost(graph):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper
    recreation_cost(v) = v['meta_freq'] * sum[e['execution_time'] for e in  path(v_0, v)]
    :type graph: nx.DiGraph
    """
    recreation_costs = {node: -1 for node in graph.nodes}
    total_weighted_cost = 0.0
    for n in nx.topological_sort(graph):
        if graph.nodes[n]['root']:
            recreation_costs[n] = 0
        else:
            cost = 0.0
            for source, _, exec_time in graph.in_edges(n, data='execution_time'):
                if recreation_costs[source] == -1:
                    raise Exception('The partial cost of the node {} should have been computed'.format(source))
                else:
                    cost += recreation_costs[source] + exec_time
            recreation_costs[n] = cost
            total_weighted_cost += (graph.nodes[n]['meta_freq'] * cost)

    for n in graph.nodes(data=True):
        n[1]['recreation_cost'] = recreation_costs[n[0]]
        n[1]['n_recreation_cost'] = (n[1]['meta_freq'] * n[1]['recreation_cost']) / total_weighted_cost


def compute_vertex_potential(graph):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper
    TODO we should make sure non-predictive models are treated properly
    TODO we should think on how to capture the potential of testing nodes
    :type graph: nx.DiGraph

    """
    ml_models = []
    potentials = {}
    # Preprocessing step: for every node in the graph if it is a predictive model assign its score as its potential
    for node in graph.nodes(data=True):
        if node[1]['type'] == 'SK_Model':
            if node[1]['score'] > 0.0:
                potentials[node[0]] = node[1]['score']
                ml_models.append(node[0])
            else:
                potentials[node[0]] = node[1]['score']
        else:
            potentials[node[0]] = 0.0

    # TODO so far the only edges going out of a model are the feature importance operation and score operation which has
    # two levels
    for m in ml_models:
        for _, out in graph.out_edges(m):
            potentials[out] = potentials[m]
            for _, outout in graph.out_edges(out):
                potentials[outout] = potentials[m]
    total_score = 0.0  # for keeping track of the sum of score to compute the score for out of reach nodes
    for n in reversed(list(nx.topological_sort(graph))):
        current_score = potentials[n]
        if current_score > 0:
            # The node is a ml model, direct evaluation node or a direct test node
            total_score += current_score
        else:
            best_potential = -1
            terminal = True
            for _, destination in graph.out_edges(n):
                terminal = False
                neighbor_potential = potentials[destination]
                if neighbor_potential >= best_potential:
                    best_potential = neighbor_potential
            if best_potential == -1 and not terminal:
                raise Exception('something went wrong, the node {} has no neighbors and is not a terminal node')
            elif best_potential == -1:
                potentials[n] = 0
            else:
                potentials[n] = best_potential
                total_score += best_potential

    for n in graph.nodes(data=True):
        n[1]['potential'] = potentials[n[0]]
        n[1]['n_potential'] = potentials[n[0]] / total_score


def cost(graph, source, destination):
    return graph.edges[source, destination]['execution_time']
