import warnings
from typing import Optional

import graphviz
import networkx as nx
import pandas as pd


def outranking_to_graph(
    outranking_matrix: pd.DataFrame,
    transitive_reduction: bool = True,
) -> Optional[nx.DiGraph]:
    """Constructs graph from outranking matrix.

    :param outranking_matrix: given outranking matrix
    :param transitive_reduction: enable transitive reduction, defaults to True

    :return: Graph created from outranking matrix if possible,
    otherwise returns None.
    """
    graph = nx.DiGraph()
    for alt_name_a in outranking_matrix.index:
        for alt_name_b in outranking_matrix.index:
            if (
                outranking_matrix.loc[alt_name_a][alt_name_b] == 1
                and alt_name_a != alt_name_b
            ):
                graph.add_edge(alt_name_a, alt_name_b)

    try:
        return nx.transitive_reduction(graph) if transitive_reduction else graph
    except nx.NetworkXError:
        warnings.warn("Directed Acyclic Graph required for transitive_reduction")


def _networkx_graph_to_graphviz(graph: nx.DiGraph) -> graphviz.Digraph:
    """Transforms NetworkX DiGraph to graphviz Digraph

    :param graph: NetworkX DiGraph

    :return: Graphviz Digraph
    """
    new_graph = graphviz.Digraph("graph", strict=True)
    new_graph.attr("node", shape="box")
    for alt_name_a, alt_name_b in graph.edges:
        new_graph.edge(alt_name_a, alt_name_b)
    return new_graph


def plot_outranking(
    outranking_matrix: pd.DataFrame, transitive_reduction: bool = True
) -> Optional[graphviz.Digraph]:
    """Creates graph plot from outranking matrix

    :param outranking_matrix: given outranking matrix
    :param transitive_reduction: enable transitive reduction, defaults to True

    :return: Graph created from outranking matrix if possible,
    otherwise returns None.
    """
    graph = outranking_to_graph(outranking_matrix, transitive_reduction)
    if graph is not None:
        return _networkx_graph_to_graphviz(graph)
