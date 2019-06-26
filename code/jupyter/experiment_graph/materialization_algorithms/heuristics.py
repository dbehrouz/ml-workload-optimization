from experiment_graph.graph.execution_graph import HistoryGraph
import networkx as nx


def compute_recreation_cost(history_graph):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper
    recreation_cost(v) = v['freq'] * sum[e['execution_time'] for e in  path(v_0, v)]
    :type history_graph: HistoryGraph
    """
    graph = history_graph.graph
    recreation_cost = {node: -1 for node in graph.nodes}
    for n in nx.topological_sort(graph):
        if graph[n]['root']:
            recreation_cost[n] = 0
        elif graph[n]['type'] == 'SuperNode':
            recreation_cost[n] = 0
        else:
            cost = 0.0
            for source, _, exec_time in graph.in_edges(n, data='execution_time'):
                if recreation_cost[source] == -1:
                    raise Exception('The partial cost of the node {} should have been computed'.format(source))
                else:
                    cost += recreation_cost[source] + exec_time
            recreation_cost[n] = cost
    return recreation_cost


def compute_vertex_potential(history_graph):
    """
    computes the recreation cost of every vertex in the graph according to formula in the paper

    :type history_graph: HistoryGraph
    """
    pass
