"""This module implements modules to explore outranking with choice methods."""
from typing import Any, Iterable, List

import pandas as pd

from .. import exceptions
from .._validation import _check_index_value_binary, _consistent_df_indexing


def _change_to_series(crisp_outranking_table: pd.DataFrame) -> pd.Series:
    _consistent_df_indexing(crisp_outranking_table=crisp_outranking_table)
    for column_name in crisp_outranking_table.columns.values:
        for row_name in crisp_outranking_table.index.values:
            _check_index_value_binary(
                crisp_outranking_table[column_name][row_name],
                name="crisp outranking relation",
            )
    return pd.Series(
        {
            alt_name_b: [
                alt_name_a
                for alt_name_a in crisp_outranking_table.index
                if crisp_outranking_table.loc[alt_name_b][alt_name_a] != 0
            ]
            for alt_name_b in crisp_outranking_table.index.values
        }
    )


def _strongly_connected_components(graph: pd.Series) -> List[List[Any]]:
    index_counter = [0]
    stack, result = [], []
    lowlink, index = {}, {}

    # Function checks if node make with another strongly_connected_component. If so
    # return list of nodes. Otherwise return only this node as a list.
    def _strong_connect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)

        successors = graph[node]
        for successor in successors:
            if successor not in index:
                _strong_connect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node], index[successor])

        if lowlink[node] == index[node]:
            connected_component = []

            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node:
                    break
            result.append(connected_component)

    try:
        for node in graph.index.values:
            if node not in index:
                _strong_connect(node)
    except KeyError as exc:
        raise exceptions.GraphError(
            "Graph contain an arc directed to a non-existent vertex "
            f"(at least one values from: {graph[node]} doesn't exists in "
            f"a {pd.Series.__name__} index values)."
        ) from exc
    except (TypeError, AttributeError) as exc:
        if "node" in locals():
            exc.args = (
                f"Successors of a vertex inside graph should be iterable, but for node: {node} "
                f"got {type(graph[node]).__name__}.",
            )
        else:
            exc.args = (
                f"Wrong graph type. Expected {pd.Series.__name__}, but "
                f"got {type(graph).__name__} instead.",
            )
        raise
    return result


def aggregate(graph: pd.Series) -> pd.Series:
    try:
        new_graph = graph.copy()
    except AttributeError as exc:
        raise TypeError(
            f"Wrong graph type. Expected {pd.Series.__name__}, "
            f"but got {type(graph).__name__} instead."
        ) from exc

    for vertices in _strongly_connected_components(graph):
        if len(vertices) == 1:
            continue
        aggregated = ", ".join(str(v) for v in vertices)
        new_connections = list(
            set([v for key in vertices for v in graph[key] if v not in vertices])
        )
        new_graph = new_graph.drop(labels=vertices)
        for key in new_graph.index.values:
            for vertex in new_graph[key][:]:
                if vertex in vertices:
                    new_graph[key].remove(vertex)
                    if aggregated not in new_graph[key]:
                        new_graph[key].append(aggregated)
        new_graph[aggregated] = new_connections
    for key in new_graph.index.values:
        if key in new_graph[key]:
            new_graph[key].remove(key)
    return new_graph


def find_vertices_without_predecessor(graph: pd.Series, **kwargs) -> List[Any]:
    if "validated" not in kwargs:
        try:
            vertex_set = set(graph.keys())
            for successor_list in graph.values:
                if not isinstance(successor_list, Iterable):
                    raise TypeError(
                        "Successor list of a graph vertex must be iterable, "
                        f"but got {type(successor_list).__name__} instead."
                    )

                if set(successor_list) - vertex_set:
                    raise exceptions.GraphError(
                        "Graph contain an arc directed to a non-existent vertex "
                        f"(at least one values from: {successor_list} doesn't exists in "
                        f"a {pd.Series.__name__} index values)."
                    )
        except (TypeError, AttributeError) as exc:
            raise TypeError(
                f"Wrong graph type. Expected {pd.Series.__name__}, "
                f"but got {type(graph).__name__} instead."
            ) from exc

    vertices_with_predecessor = list(
        set([v for key in graph.index.values for v in graph[key]])
    )
    return [vertex for vertex in graph.index if vertex not in vertices_with_predecessor]


def find_kernel(crisp_outranking_table: pd.DataFrame) -> List[str]:
    """This function finds a kernel (out1) in a graph
    constructed on the basis of a crisp outranking relation
    :param crisp_outranking_table: table with crisp relations
    between alternatives
    :return: every alternative that is in kernel
    """
    graph = _change_to_series(crisp_outranking_table)
    graph = aggregate(graph)
    not_kernel: List = []
    kernel = find_vertices_without_predecessor(graph, validated=True)

    for vertex in kernel:
        not_kernel += graph.pop(vertex)

    while len(graph.keys()) != 0:
        vertices = find_vertices_without_predecessor(graph, validated=True)
        for vertex in vertices:
            if vertex not in not_kernel:
                kernel.append(vertex)
                not_kernel += graph[vertex]
            graph.pop(vertex)

    return kernel
