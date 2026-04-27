"""High-level planner assembly entry point."""

from __future__ import annotations

from pathlib import Path

from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node
from MAPS.core.layer import Layer, LayerInput, LayerOutput
from MAPS.core.pipeline import Pipeline
from MAPS.core.stage import Stage
from MAPS.core.transition import Transition
from MAPS.importers.onnx.importer import import_onnx_graph
from MAPS.planner.select_stage import select_stages
from MAPS.transitions import build_transition
from MAPS.planner.spatial_mapping import map_spatially, place_stage_plans
from MAPS.planner.workload_balancing import StagePlan, balance_workload


def build_pipeline(
    model_path: str | Path,
    mesh: Mesh,
    print_workload_balancing: bool = False,
    print_spatial_mapping: bool = False,
    print_spatial_mapping_progress: bool = False,
    enable_lossless_spatial_mapping_pruning: bool = False,
    enable_lossy_spatial_mapping_pruning: bool = False,
    require_l2_input_access_point: bool = False,
    require_l2_output_access_point: bool = False,
) -> Pipeline:
    """Build a pipeline plan from one ONNX model."""

    graph = import_onnx_graph(model_path)
    stage_selection = select_stages(graph)
    stage_plans = balance_workload(
        graph,
        mesh,
        debug=print_workload_balancing,
        stage_selection=stage_selection,
    )
    mapping = map_spatially(
        graph,
        mesh,
        stage_plans,
        enable_lossless_pruning=enable_lossless_spatial_mapping_pruning,
        max_placements_per_stage=16 if enable_lossy_spatial_mapping_pruning else None,
        show_progress=print_spatial_mapping_progress,
        print_costs=print_spatial_mapping,
        require_l2_input_access_point=require_l2_input_access_point,
        require_l2_output_access_point=require_l2_output_access_point,
    )
    placed_plans = place_stage_plans(stage_plans, mapping)
    return _build_pipeline_from_graph(graph, mesh, placed_plans)


def _build_pipeline_from_graph(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
) -> Pipeline:
    stage_selection = _resolve_stage_selection(graph, stage_plans)
    node_stage_ids = {
        id(node): stage_id
        for stage_id, stage_nodes in stage_selection.items()
        for node in stage_nodes
    }
    node_layer_ids = {
        id(node): layer_idx
        for stage_nodes in stage_selection.values()
        for layer_idx, node in enumerate(stage_nodes)
    }
    tensor_id_by_tensor = {tensor: tensor_id for tensor_id, tensor in enumerate(graph.tensors)}
    producer_by_tensor = {
        tensor: node
        for node in graph.nodes
        for tensor in node.outputs
    }

    transitions: list[Transition] = []
    transition_id_by_stage_layer_input: dict[tuple[int, int, int], int] = {}

    for dst_stage_id, stage_nodes in stage_selection.items():
        dst_plan = stage_plans[dst_stage_id]
        for dst_layer_idx, dst_node in enumerate(stage_nodes):
            for dst_input_idx, tensor in enumerate(dst_node.inputs):
                src_node = producer_by_tensor.get(tensor)
                if src_node is None:
                    continue

                src_stage_id = node_stage_ids[id(src_node)]
                if src_stage_id == dst_stage_id:
                    continue

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
                        src_layout=_node_output_layout(src_plan, src_node)[src_output_idx],
                        dst_layout=_node_input_layout(dst_plan, dst_node)[dst_input_idx],
                    )
                )
                transition_id_by_stage_layer_input[(dst_stage_id, dst_layer_idx, dst_input_idx)] = transition_id

    stages = tuple(
        _build_stage(
            stage_id=stage_id,
            plan=stage_plans[stage_id],
            stage_nodes=stage_selection[stage_id],
            producer_by_tensor=producer_by_tensor,
            node_stage_ids=node_stage_ids,
            node_layer_ids=node_layer_ids,
            tensor_id_by_tensor=tensor_id_by_tensor,
            transition_id_by_stage_layer_input=transition_id_by_stage_layer_input,
        )
        for stage_id in stage_selection
    )

    return Pipeline(
        name=graph.name,
        mesh=mesh,
        tensors=graph.tensors,
        stages=stages,
        transitions=tuple(transitions),
    )


def _build_stage(
    stage_id: int,
    plan: StagePlan,
    stage_nodes: tuple[Node, ...],
    producer_by_tensor: dict[object, Node],
    node_stage_ids: dict[int, int],
    node_layer_ids: dict[int, int],
    tensor_id_by_tensor: dict[object, int],
    transition_id_by_stage_layer_input: dict[tuple[int, int, int], int],
) -> Stage:
    layers = []
    for layer_idx, node in enumerate(stage_nodes):
        inputs = tuple(
            LayerInput(
                tensor_id=tensor_id_by_tensor[tensor],
                source=_input_source(
                    tensor_id=tensor_id_by_tensor[tensor],
                    producer=producer_by_tensor.get(tensor),
                    stage_id=stage_id,
                    layer_idx=layer_idx,
                    node_stage_ids=node_stage_ids,
                    node_layer_ids=node_layer_ids,
                    transition_id=transition_id_by_stage_layer_input.get((stage_id, layer_idx, input_idx)),
                ),
            )
            for input_idx, tensor in enumerate(node.inputs)
        )
        outputs = tuple(
            LayerOutput(
                tensor_id=tensor_id_by_tensor[tensor],
                layout=_node_output_layout(plan, node)[output_idx],
            )
            for output_idx, tensor in enumerate(node.outputs)
        )
        layers.append(Layer(node=node, inputs=inputs, outputs=outputs))

    return Stage(
        name=_stage_name(stage_nodes),
        submesh=_stage_submesh(plan),
        layers=tuple(layers),
    )


def _input_source(
    tensor_id: int,
    producer: Node | None,
    stage_id: int,
    layer_idx: int,
    node_stage_ids: dict[int, int],
    node_layer_ids: dict[int, int],
    transition_id: int | None,
):
    if producer is not None and node_stage_ids[id(producer)] == stage_id:
        return LayerInput.local(
            tensor_id=tensor_id,
            layer_idx=node_layer_ids[id(producer)],
        ).source
    if transition_id is None:
        return LayerInput.external(tensor_id=tensor_id, base_addr=tensor_id + 1).source
    return LayerInput.transition(tensor_id=tensor_id, transition_id=transition_id).source


def _resolve_stage_selection(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
) -> dict[int, tuple[Node, ...]]:
    """Return selected stages using plan metadata when available."""

    if all(plan.nodes for plan in stage_plans.values()):
        return {
            stage_id: plan.nodes
            for stage_id, plan in stage_plans.items()
        }
    return {
        stage_id: (node,)
        for stage_id, node in enumerate(graph.nodes)
    }


def _stage_name(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact generated stage name."""

    return "+".join(node.name for node in stage_nodes)


def _stage_submesh(plan: StagePlan):
    """Return the concrete submesh attached to one stage plan."""

    if plan.node_output_layouts:
        for layouts in plan.node_output_layouts:
            if layouts:
                return layouts[0].submesh
    if plan.node_input_layouts:
        for layouts in plan.node_input_layouts:
            if layouts:
                return layouts[0].submesh
    if plan.output_layouts:
        return plan.output_layouts[0].submesh
    if plan.input_layouts:
        return plan.input_layouts[0].submesh
    raise ValueError(f"stage {plan.stage_id} has no layouts bound to a submesh")


def _node_input_layout(plan: StagePlan, node: Node):
    """Return one node's input layouts from a stage plan."""

    if plan.node_input_layouts:
        return plan.node_input_layouts[_plan_node_index(plan, node)]
    return plan.input_layouts


def _node_output_layout(plan: StagePlan, node: Node):
    """Return one node's output layouts from a stage plan."""

    if plan.node_output_layouts:
        return plan.node_output_layouts[_plan_node_index(plan, node)]
    return plan.output_layouts


def _plan_node_index(plan: StagePlan, node: Node) -> int:
    """Return one node's index inside a grouped stage plan."""

    for node_idx, candidate in enumerate(plan.nodes):
        if candidate is node:
            return node_idx
    raise ValueError(f"node {node.name} is not present in stage plan {plan.stage_id}")


def _node_output_index(node: Node, tensor: object) -> int:
    for output_idx, candidate in enumerate(node.outputs):
        if candidate == tensor:
            return output_idx
    raise ValueError(f"tensor is not an output of node {node.name}")
