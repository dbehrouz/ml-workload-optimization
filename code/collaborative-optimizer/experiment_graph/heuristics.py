import networkx as nx


def compute_recreation_cost(graph):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper
    recreation_cost(v) = v['meta_freq'] * sum[e['execution_time'] for e in  path(v_0, v)]
    :type graph: nx.DiGraph
    """
    recreation_cost = {node: -1 for node in graph.nodes}
    for n in nx.topological_sort(graph):
        if graph.nodes[n]['root']:
            recreation_cost[n] = 0
        else:
            cost = 0.0
            for source, _, exec_time in graph.in_edges(n, data='execution_time'):
                if recreation_cost[source] == -1:
                    raise Exception('The partial cost of the node {} should have been computed'.format(source))
                else:
                    cost += recreation_cost[source] + exec_time
            recreation_cost[n] = cost

    for n in graph.nodes(data=True):
        n[1]['recreation_cost'] = recreation_cost[n[0]]

    return recreation_cost


def compute_vertex_potential(graph):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper
    TODO we should make sure non-predictive models are treated properly
    TODO we should think on how to capture the potential of testing nodes
    :type graph: nx.DiGraph
    :type modify_graph: bool

    """
    potentials = {}
    ml_models = []
    # A dictionary of the form 'vertex_id' = (best_model_quality, cost_to_best_model)
    cost_to_best_model = {}
    # Preprocessing step: for every node in the graph if it is a predictive model assign its score as its potential
    for node in graph.nodes(data=True):
        if node[1]['type'] == 'SK_Model':
            potentials[node[0]] = node[1]['score']
            if node[1]['score'] > 0.0:
                ml_models.append(node[0])
                cost_to_best_model[node[0]] = (node[1]['score'], 0.0)
        elif graph.out_degree(node[0]) == 0:
            potentials[node[0]] = -1.0
        else:
            potentials[node[0]] = 0.0

    # TODO so far the only edges going out of a model are the feature importance operation and score operation which has
    # two levels
    for m in ml_models:
        for _, out in graph.out_edges(m):
            potentials[out] = potentials[m]
            for _, outout in graph.out_edges(out):
                potentials[outout] = potentials[m]
                cost_to_best_model[out] = (graph.nodes[m]['score'], 0.0)

    total_score = 0.0  # for keeping track of the sum of score to compute the score for out of reach nodes
    for n in reversed(list(nx.topological_sort(graph))):
        current_score = potentials[n]
        if current_score > 0:
            # The node is a ml model, direct evaluation node or a direct test node
            total_score += current_score
        else:
            best_potential_among_neighbors = -1
            neighbor_with_largest_potential = -1
            terminal = True
            for _, destination in graph.out_edges(n):
                terminal = False
                neighbor_potential = potentials[destination]
                if neighbor_potential >= best_potential_among_neighbors:
                    neighbor_with_largest_potential = destination
                    best_potential_among_neighbors = neighbor_potential
            if neighbor_with_largest_potential == -1 and not terminal:
                raise Exception('something went wrong, the node {} has no neighbors and is not a terminal node')
            if best_potential_among_neighbors == -1:
                potentials[n] = -1
                continue
            model_score, cost_to_model = cost_to_best_model[neighbor_with_largest_potential]

            my_cost = (cost_to_model + cost(graph, n, neighbor_with_largest_potential))
            if my_cost == 0.0:
                my_potential = model_score
            else:
                my_potential = model_score / my_cost
            cost_to_best_model[n] = (model_score, my_cost)
            potentials[n] = my_potential
            total_score += my_potential

    default = total_score / len(graph.nodes)
    for k, v in potentials.iteritems():
        if v == -1:
            potentials[k] = default

    for n in graph.nodes(data=True):
        n[1]['potential'] = potentials[n[0]]

    return potentials


def cost(graph, source, destination):
    return graph.edges[source, destination]['execution_time']
