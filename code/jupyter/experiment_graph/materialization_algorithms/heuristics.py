import networkx as nx


def compute_recreation_cost(graph):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper
    recreation_cost(v) = v['meta_freq'] * sum[e['execution_time'] for e in  path(v_0, v)]
    :type graph: input history graph
    """
    partial_recreation_cost = {node: -1 for node in graph.nodes}
    for n in nx.topological_sort(graph):
        if graph[n]['root']:
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

    return recreation_cost


def compute_vertex_potential(graph):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper

    :type graph: input history graph
    """
    pass
