import networkx as nx


def compute_recreation_cost(graph, modify_graph=True):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper
    recreation_cost(v) = v['meta_freq'] * sum[e['execution_time'] for e in  path(v_0, v)]
    :type modify_graph: bool
    :type graph: nx.DiGraph
    """
    partial_recreation_cost = {node: -1 for node in graph.nodes}
    for n in nx.topological_sort(graph):
        if graph.nodes[n]['root']:
            partial_recreation_cost[n] = 0
        else:
            cost = 0.0
            for source, _, exec_time in graph.in_edges(n, data='execution_time'):
                if partial_recreation_cost[source] == -1:
                    raise Exception('The partial cost of the node {} should have been computed'.format(source))
                else:
                    cost += partial_recreation_cost[source] + exec_time
            partial_recreation_cost[n] = cost
    recreation_cost = {}
    for n in graph.nodes(data='meta_freq'):
        recreation_cost[n[0]] = n[1] * partial_recreation_cost[n[0]]
    if modify_graph:
        for n in graph.nodes(data=True):
            n[1]['recreation_cost'] = recreation_cost[n[0]]

    return recreation_cost


def compute_vertex_potential(graph, modify_graph=True, alpha=0.9):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper
    TODO we should make sure non-predictive models are treated properly
    TODO we should think on how to capture the potential of testing nodes
    :type alpha: float damping factor, default=0.9
    :type graph: nx.DiGraph
    :type modify_graph: bool

    """
    potentials = {}
    connected_models = {}
    ml_models = []
    for node in graph.nodes(data=True):
        if node[1]['type'] == 'SK_Model':
            potentials[node[0]] = node[1]['score']
            if node[1]['score'] > 0.0:
                connected_models[node[0]] = {node[0]}
                ml_models.append(node[0])
            else:
                connected_models[node[0]] = set()
        elif graph.out_degree(node[0]) == 0:
            potentials[node[0]] = -1.0
            connected_models[node[0]] = set()
        else:
            potentials[node[0]] = 0.0
            connected_models[node[0]] = set()

    # TODO this should be investigated, but it only seems fair if we assign potential and num pipeline to evaluation
    # and test datasets as well
    for m in ml_models:
        out = list(graph.out_edges(m))
        assert len(out) == 1
        out = out[0][1]
        connected_models[out] = {m}
        potentials[out] = potentials[m]
        outout = list(graph.out_edges(out))
        assert len(outout) == 1
        outout = outout[0][1]
        connected_models[outout] = {m}
        potentials[outout] = potentials[m]

    total_score = 0.0  # for keeping track of the sum of score to compute the score for out of reach nodes
    for n in reversed(list(nx.topological_sort(graph))):
        current_score = potentials[n]
        if current_score > 0:
            # The node is a ml model
            total_score += current_score
        else:
            best_score_among_neighbors = -1
            selected_node = -1
            terminal = True
            models = set()
            for _, destination in graph.out_edges(n):
                if destination in ml_models:
                    models.add(destination)
                else:
                    for e in connected_models[destination]:
                        models.add(e)
                terminal = False
                score = potentials[destination]
                if score >= best_score_among_neighbors:
                    selected_node = destination
                    best_score_among_neighbors = score
            if selected_node == -1 and not terminal:
                raise Exception('something went wrong, the node {} has no neighbors and is not a terminal node')
            if best_score_among_neighbors == -1:
                potentials[n] = -1
                continue
            s = alpha * best_score_among_neighbors
            potentials[n] = s
            total_score += s
            connected_models[n] = models

    default = total_score / len(graph.nodes)
    for k, v in potentials.iteritems():
        if v == -1:
            potentials[k] = default
    num_pipelines = {}
    if modify_graph:
        for n in graph.nodes(data=True):
            n[1]['potential'] = potentials[n[0]]
            length = len(connected_models[n[0]])
            n[1]['num_pipelines'] = length
            num_pipelines[n[0]] = length

    return potentials, num_pipelines
