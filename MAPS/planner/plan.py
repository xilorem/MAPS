"""High-level planner assembly entry point."""

from __future__ import annotations

from pathlib import Path

from MAPS.arch import Mesh
from MAPS.builders.transition_builder import build_transition
from MAPS.core.graph import Graph, Node
from MAPS.core.layer import Layer, LayerInput, LayerOutput
from MAPS.core.pipeline import Pipeline
from MAPS.core.stage import Stage
from MAPS.core.transition import Transition
from MAPS.importers.onnx.importer import import_onnx_graph
from MAPS.planner.spatial_mapping import map_spatially, place_stage_plans
from MAPS.planner.workload_balancing import StagePlan, balance_stage_plans


def build_pipeline(model_path: str | Path, mesh: Mesh) -> Pipeline:
    """Build a pipeline plan from one ONNX model."""

    graph = import_onnx_graph(model_path)
    stage_plans = balance_stage_plans(graph, mesh)
    mapping = map_spatially(graph, mesh, stage_plans)
    placed_plans = place_stage_plans(stage_plans, mapping)
    return _build_pipeline_from_graph(graph, mesh, placed_plans)


def _build_pipeline_from_graph(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
) -> Pipeline:
    tensor_id_by_tensor = {tensor: tensor_id for tensor_id, tensor in enumerate(graph.tensors)}
    producer_by_tensor = {
        tensor: stage_id
        for stage_id, node in enumerate(graph.nodes)
        for tensor in node.outputs
    }

    transitions: list[Transition] = []
    transition_id_by_stage_input: dict[tuple[int, int], int] = {}

    for dst_stage_id, dst_node in enumerate(graph.nodes):
        dst_plan = stage_plans[dst_stage_id]
        for dst_input_idx, tensor in enumerate(dst_node.inputs):
            if tensor not in producer_by_tensor:
                continue

            src_stage_id = producer_by_tensor[tensor]
            src_node = graph.nodes[src_stage_id]
            src_plan = stage_plans[src_stage_id]
            src_output_idx = _node_output_index(src_node, tensor)
            transition_id = len(transitions)
            transitions.append(
                build_transition(
                    name=f"transition_{src_node.name}_to_{dst_node.name}_{tensor.name}",
                    tensor=tensor,
                    tensor_id=tensor_id_by_tensor[tensor],
                    src_layer_id=src_stage_id,
                    src_output_idx=src_output_idx,
                    dst_layer_id=dst_stage_id,
                    dst_input_idx=dst_input_idx,
                    src_layout=src_plan.output_layouts[src_output_idx],
                    dst_layout=dst_plan.input_layouts[dst_input_idx],
                )
            )
            transition_id_by_stage_input[(dst_stage_id, dst_input_idx)] = transition_id

    stages = tuple(
        _build_stage(
            graph=graph,
            stage_id=stage_id,
            plan=stage_plans[stage_id],
            tensor_id_by_tensor=tensor_id_by_tensor,
            transition_id_by_stage_input=transition_id_by_stage_input,
        )
        for stage_id in range(len(graph.nodes))
    )

    return Pipeline(
        name=graph.name,
        mesh=mesh,
        tensors=graph.tensors,
        stages=stages,
        transitions=tuple(transitions),
    )


def _build_stage(
    graph: Graph,
    stage_id: int,
    plan: StagePlan,
    tensor_id_by_tensor: dict[object, int],
    transition_id_by_stage_input: dict[tuple[int, int], int],
) -> Stage:
    node = graph.nodes[stage_id]
    inputs = tuple(
        LayerInput(
            tensor_id=tensor_id_by_tensor[tensor],
            source=_input_source(
                tensor_id=tensor_id_by_tensor[tensor],
                transition_id=transition_id_by_stage_input.get((stage_id, input_idx)),
            ),
        )
        for input_idx, tensor in enumerate(node.inputs)
    )
    outputs = tuple(
        LayerOutput(
            tensor_id=tensor_id_by_tensor[tensor],
            layout=plan.output_layouts[output_idx],
        )
        for output_idx, tensor in enumerate(node.outputs)
    )
    return Stage(
        name=node.name,
        submesh=plan.output_layouts[0].submesh,
        layers=(Layer(node=node, inputs=inputs, outputs=outputs),),
    )


def _input_source(
    tensor_id: int,
    transition_id: int | None,
):
    if transition_id is None:
        return LayerInput.external(tensor_id=tensor_id, base_addr=tensor_id + 1).source
    return LayerInput.transition(tensor_id=tensor_id, transition_id=transition_id).source


def _node_output_index(node: Node, tensor: object) -> int:
    for output_idx, candidate in enumerate(node.outputs):
        if candidate == tensor:
            return output_idx
    raise ValueError(f"tensor is not an output of node {node.name}")
