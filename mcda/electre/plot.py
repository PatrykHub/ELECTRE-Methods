"""This module implements methods to make a visualization of an outranking relation."""
import warnings
from typing import Optional

import graphviz
import networkx as nx
import pandas as pd


def outranking_to_graph(
    outranking_matrix: pd.DataFrame,
    transitive_reduction: bool = True,
) -> Optional[nx.DiGraph]:
    """Constructs graph from outranking matrix (if it is possible to create DAG).

    :param outranking_matrix: given outranking matrix
    :param transitive_reduction: enable transitive reduction, defaults to True

    :return: Graph created from outranking matrix if possible,
        otherwise displays warning and returns None.
    """
    graph = nx.DiGraph()
    for alt_name_a in outranking_matrix.index.values:
        for alt_name_b in outranking_matrix.index.values:
            if (
                outranking_matrix.loc[alt_name_a][alt_name_b] == 1
                and alt_name_a != alt_name_b
            ):
                graph.add_edge(alt_name_a, alt_name_b)

    try:
        return nx.transitive_reduction(graph) if transitive_reduction else graph
    except nx.NetworkXError:
        warnings.warn("Directed Acyclic Graph required for transitive_reduction")
    return None


def _networkx_graph_to_graphviz(graph: nx.DiGraph) -> graphviz.Digraph:
    """Transforms networkx DiGraph to graphviz Digraph.

    :param graph: networkx DiGraph

    :return: graphviz Digraph
    """
    new_graph = graphviz.Digraph("graph", strict=True)
    new_graph.attr("node", shape="box")
    for alt_name_a, alt_name_b in graph.edges:
        new_graph.edge(alt_name_a, alt_name_b)
    return new_graph


def plot_outranking(
    outranking_matrix: pd.DataFrame, transitive_reduction: bool = True
) -> Optional[graphviz.Digraph]:
    """Creates graph plot from outranking matrix (if it is possible to create DAG).

    :param outranking_matrix: given outranking matrix
    :param transitive_reduction: enable transitive reduction, defaults to True

    :return: Graph created from outranking matrix if possible,
        otherwise displays warning and returns None.
    """
    graph = outranking_to_graph(outranking_matrix, transitive_reduction)
    if graph is not None:
        return _networkx_graph_to_graphviz(graph)
    return None
