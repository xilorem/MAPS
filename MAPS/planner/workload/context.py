"""Validated inputs and graph classifications for workload balancing."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.graph import Graph
from MAPS.planner.contracts.stages import StageSelection


@dataclass(frozen=True)
class WorkloadContext:
    """Graph facts shared by allocation and bottleneck estimation."""

    graph: Graph
    stage_selection: StageSelection
    initializer_tensors: frozenset
    graph_inputs: frozenset
    graph_outputs: frozenset
    producer_stage_id_by_tensor: dict[object, int]


def build_workload_context(
    graph: Graph,
    stage_selection: StageSelection,
) -> WorkloadContext:
    """Validate stage coverage and classify boundary and produced tensors."""

    resolved_selection = resolve_stage_selection(graph, stage_selection)
    initializer_tensors = frozenset(graph.initializers)
    return WorkloadContext(
        graph=graph,
        stage_selection=resolved_selection,
        initializer_tensors=initializer_tensors,
        graph_inputs=frozenset(graph.inputs) - initializer_tensors,
        graph_outputs=frozenset(graph.outputs),
        producer_stage_id_by_tensor=producer_stage_id_by_tensor(resolved_selection),
    )


def resolve_stage_selection(
    graph: Graph,
    stage_selection: StageSelection,
) -> StageSelection:
    """Return an explicit selection that covers every graph node exactly once."""

    graph_node_ids = {id(node) for node in graph.nodes}
    selected_node_ids: set[int] = set()
    resolved: StageSelection = {}
    for stage_id, stage_nodes in stage_selection.items():
        if not stage_nodes:
            raise ValueError(f"stage {stage_id} must contain at least one node")
        for node in stage_nodes:
            node_id = id(node)
            if node_id not in graph_node_ids:
                raise ValueError(
                    f"stage {stage_id} contains node {node.name} "
                    f"not present in graph {graph.name}"
                )
            if node_id in selected_node_ids:
                raise ValueError(
                    f"node {node.name} appears in more than one selected stage"
                )
            selected_node_ids.add(node_id)
        resolved[stage_id] = tuple(stage_nodes)

    if selected_node_ids != graph_node_ids:
        missing = tuple(
            node.name
            for node in graph.nodes
            if id(node) not in selected_node_ids
        )
        raise ValueError(f"selected stages do not cover all graph nodes, missing={missing}")
    return resolved


def producer_stage_id_by_tensor(
    stage_selection: StageSelection,
) -> dict[object, int]:
    """Map every produced tensor to the id of its selected producer stage."""

    return {
        tensor: stage_id
        for stage_id, stage_nodes in stage_selection.items()
        for node in stage_nodes
        for tensor in node.outputs
    }
