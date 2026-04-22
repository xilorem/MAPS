"""Graph-level ONNX parsing orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from MAPS.core.graph import Graph

from .node_parser import parse_node
from .ops.softmax import lower_softmax_node
from .tensor_parser import collect_scheduler_tensors
from .utils import build_lowered_graph_edges

if TYPE_CHECKING:
    from onnx import GraphProto


def parse_graph(graph: "GraphProto", *, graph_name: str | None = None) -> Graph:
    """Parse one ONNX graph into the shared scheduler graph IR."""

    tensors = collect_scheduler_tensors(graph)
    nodes = []
    for node_idx, node in enumerate(graph.node):
        if node.op_type == "Softmax":
            new_tensors, lowered_nodes = lower_softmax_node(node, node_idx, tensors)
            for tensor in new_tensors:
                if tensor.name in tensors:
                    raise ValueError(f"tensor '{tensor.name}' is already present in graph metadata")
                tensors[tensor.name] = tensor
            nodes.extend(lowered_nodes)
            continue

        nodes.append(parse_node(node, node_idx, tensors))

    initializer_names = {initializer.name for initializer in graph.initializer}
    graph_input_names = {value.name for value in graph.input if value.name not in initializer_names}
    graph_output_names = tuple(value.name for value in graph.output)
    lowered_nodes = tuple(nodes)

    return Graph(
        name=graph_name or graph.name,
        tensors=tuple(tensors.values()),
        nodes=lowered_nodes,
        edges=build_lowered_graph_edges(lowered_nodes, tensors, graph_output_names),
        inputs=tuple(tensors[value.name] for value in graph.input if value.name in graph_input_names),
        outputs=tuple(tensors[value.name] for value in graph.output),
        initializers=tuple(
            tensors[initializer.name]
            for initializer in graph.initializer
            if initializer.name in tensors
        ),
    )
