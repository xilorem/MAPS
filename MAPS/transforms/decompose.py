"""Composite-op decomposition pass."""

from __future__ import annotations

from MAPS.core.graph import Graph
from MAPS.ops.registry import get_op_for_payload

from .graph_utils import build_graph_edges_from_nodes


def decompose_graph(graph: Graph) -> Graph:
    """Replace composite nodes with the primitive nodes produced by their op specs."""

    tensors = {tensor.name: tensor for tensor in graph.tensors}
    nodes = []

    for node in graph.nodes:
        spec = None if node.payload is None else get_op_for_payload(node.payload)
        if spec is None or spec.decompose is None:
            nodes.append(node)
            continue

        new_tensors, lowered_nodes = spec.decompose(node)
        for tensor in new_tensors:
            if tensor.name in tensors:
                raise ValueError(f"tensor '{tensor.name}' is already present in graph metadata")
            tensors[tensor.name] = tensor
        nodes.extend(lowered_nodes)

    lowered_nodes = tuple(nodes)
    graph_output_names = tuple(tensor.name for tensor in graph.outputs)

    return Graph(
        name=graph.name,
        tensors=tuple(tensors.values()),
        nodes=lowered_nodes,
        edges=build_graph_edges_from_nodes(lowered_nodes, tensors, graph_output_names),
        inputs=graph.inputs,
        outputs=graph.outputs,
        initializers=graph.initializers,
    )
