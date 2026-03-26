"""Graph-level ONNX parsing orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from MAPS.core.graph import Graph

from .node_parser import parse_node
from .tensor_parser import collect_scheduler_tensors
from .utils import build_graph_edges

if TYPE_CHECKING:
    from onnx import GraphProto


def parse_graph(graph: "GraphProto", *, graph_name: str | None = None) -> Graph:
    """Parse one ONNX graph into the shared scheduler graph IR."""

    tensors = collect_scheduler_tensors(graph)
    nodes = tuple(
        parse_node(node, node_idx, tensors)
        for node_idx, node in enumerate(graph.node)
    )

    initializer_names = {initializer.name for initializer in graph.initializer}
    graph_input_names = {value.name for value in graph.input if value.name not in initializer_names}

    return Graph(
        name=graph_name or graph.name,
        tensors=tuple(tensors.values()),
        nodes=nodes,
        edges=build_graph_edges(graph, nodes, tensors),
        inputs=tuple(tensors[value.name] for value in graph.input if value.name in graph_input_names),
        outputs=tuple(tensors[value.name] for value in graph.output),
        initializers=tuple(
            tensors[initializer.name]
            for initializer in graph.initializer
            if initializer.name in tensors
        ),
    )
