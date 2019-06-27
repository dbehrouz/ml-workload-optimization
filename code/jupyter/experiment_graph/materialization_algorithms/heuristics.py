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
    best_model_so_far = {}
    ml_models = []
    for node in graph.nodes(data=True):
        if node[1]['type'] == 'SK_Model':
            best_model_so_far[node[0]] = node[1]['score']
            if node[1]['score'] > 0.0:
                ml_models.append(node[0])
        elif graph.out_degree(node[0]) == 0:
            best_model_so_far[node[0]] = -1.0
        else:
            best_model_so_far[node[0]] = 0.0

    distance_to_best_model = {node: 0 for node in graph.nodes}
    for m in ml_models:
        dest = list(graph.out_edges(m))
        if len(dest) > 1:
            raise Exception('A model can only be evaluated against one dataset')
        best_model_so_far[dest[0][1]] = best_model_so_far[m]
        eval_dest = list(graph.out_edges(dest[0][1]))
        if len(eval_dest) > 1:
            raise Exception('There can only be one evaluation node for one model')
        best_model_so_far[eval_dest[0][1]] = best_model_so_far[m]

    for n in reversed(list(nx.topological_sort(graph))):
        print graph.nodes[n]['type']
        if best_model_so_far[n] != 0:
            # The node is either a ml model or a terminal node
            continue
        else:
            best_score_among_neighbors = -1
            selected_node = -1
            terminal = True
            for _, destination in graph.out_edges(n):
                terminal = False
                score = best_model_so_far[destination]
                if score >= best_score_among_neighbors:
                    selected_node = destination
                    best_score_among_neighbors = score
            if selected_node == -1 and not terminal:
                raise Exception('something went wrong, the node {} has no neighbors and is not a terminal node')
            if best_score_among_neighbors == -1:
                best_model_so_far[n] = -1
                continue

            distance = distance_to_best_model[selected_node] + 1
            best_model_so_far[n] = pow(alpha, distance) * best_score_among_neighbors
            distance_to_best_model[n] = distance

    if modify_graph:
        for n in graph.nodes(data=True):
            n[1]['potential'] = best_model_so_far[n[0]]

    return best_model_so_far
