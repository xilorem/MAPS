"""High-level planner assembly entry point for connected-submesh planning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from MAPS.arch import Mesh, Tile
from MAPS.core.graph import Graph, Node
from MAPS.core.layout import TensorSlice, tile_tensor_slice
from MAPS.importers.onnx.importer import import_onnx_graph
from MAPS.pipeline.finalization import Finalization, FinalizationFragment
from MAPS.pipeline.initialization import Initialization, InitializationFragment
from MAPS.pipeline.layer import Layer, LayerInput, LayerOutput
from MAPS.pipeline.pipeline import Pipeline
from MAPS.pipeline.stage import Stage
from MAPS.planner.workload_balancing_v2 import (
    StagePlan,
    _producer_stage_id_by_tensor,
    _worst_tile_compute_workload_for_stage,
    _worst_tile_l2_transfer_workload_for_stage,
    balance_workload,
)
from MAPS.planner.select_stage import select_stages
from MAPS.planner.spatial_mapping_v2 import map_spatially, place_stage_plans
from MAPS.transitions import build_transition
from MAPS.transitions.model import Transition
from MAPS.utils.pipeline_json import write_pipeline_json


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
    output_json_path: str | Path | None = None,
) -> Pipeline:
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
    pipeline = _build_pipeline_from_graph(graph, mesh, placed_plans)
    _print_pipeline_stage_cost(graph, mesh, placed_plans)
    if output_json_path is not None:
        write_pipeline_json(pipeline, output_json_path)
    return pipeline


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
    producer_by_tensor = {tensor: node for node in graph.nodes for tensor in node.outputs}

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
                dst_output_layouts = _node_output_layout(dst_plan, dst_node)
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
                        dst_layout=dst_output_layouts[0],
                        dst_required_slices=_transition_required_slices(
                            tensor=tensor,
                            dst_node=dst_node,
                            dst_output_layouts=dst_output_layouts,
                        ),
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
    initializations = _build_initializations(
        graph=graph,
        stage_selection=stage_selection,
        stage_plans=stage_plans,
        producer_by_tensor=producer_by_tensor,
        tensor_id_by_tensor=tensor_id_by_tensor,
    )
    finalizations = _build_finalizations(
        graph=graph,
        stage_plans=stage_plans,
        node_stage_ids=node_stage_ids,
        producer_by_tensor=producer_by_tensor,
        tensor_id_by_tensor=tensor_id_by_tensor,
    )
    return Pipeline(
        name=graph.name,
        mesh=mesh,
        tensors=tuple(
            replace(tensor, is_initializer=tensor in graph.initializers)
            for tensor in graph.tensors
        ),
        stages=stages,
        transitions=tuple(transitions),
        initializations=initializations,
        finalizations=finalizations,
    )


def _build_initializations(
    graph: Graph,
    stage_selection: dict[int, tuple[Node, ...]],
    stage_plans: dict[int, StagePlan],
    producer_by_tensor: dict[object, Node],
    tensor_id_by_tensor: dict[object, int],
) -> tuple[Initialization, ...]:
    layer_id_by_node = {id(node): layer_id for layer_id, node in enumerate(graph.nodes)}
    initializations = []
    for stage_id, stage_nodes in stage_selection.items():
        plan = stage_plans[stage_id]
        for node in stage_nodes:
            output_layouts = _node_output_layout(plan, node)
            for input_idx, tensor in enumerate(node.inputs):
                if tensor in producer_by_tensor:
                    continue
                required_slices = _transition_required_slices(
                    tensor=tensor,
                    dst_node=node,
                    dst_output_layouts=output_layouts,
                )
                initializations.append(
                    Initialization(
                        name=f"init_{tensor.name}",
                        tensor_id=tensor_id_by_tensor[tensor],
                        dst_layer_id=layer_id_by_node[id(node)],
                        dst_input_idx=input_idx,
                        fragments=tuple(
                            InitializationFragment(
                                src_hartid=-1,
                                dst_hartid=tile.tile_id,
                                src_slice=tensor_slice,
                                dst_slice=tensor_slice,
                            )
                            for tile, tensor_slice in required_slices
                        ),
                    )
                )
    return tuple(initializations)


def _build_finalizations(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
    node_stage_ids: dict[int, int],
    producer_by_tensor: dict[object, Node],
    tensor_id_by_tensor: dict[object, int],
) -> tuple[Finalization, ...]:
    layer_id_by_node = {id(node): layer_id for layer_id, node in enumerate(graph.nodes)}
    finalizations = []
    for tensor in graph.outputs:
        src_node = producer_by_tensor[tensor]
        src_output_idx = _node_output_index(src_node, tensor)
        src_layout = _node_output_layout(stage_plans[node_stage_ids[id(src_node)]], src_node)[src_output_idx]
        finalizations.append(
            Finalization(
                name=f"output_{tensor.name}",
                tensor_id=tensor_id_by_tensor[tensor],
                src_layer_id=layer_id_by_node[id(src_node)],
                src_output_idx=src_output_idx,
                fragments=tuple(
                    FinalizationFragment(
                        src_hartid=tile.tile_id,
                        dst_hartid=-1,
                        src_slice=tile_tensor_slice(tensor, src_layout, tile),
                        dst_slice=tile_tensor_slice(tensor, src_layout, tile),
                    )
                    for tile in src_layout.submesh.tiles
                ),
            )
        )
    return tuple(finalizations)


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
    return Stage(
        name=_stage_name(stage_nodes),
        submesh=_stage_submesh(plan),
        layers=tuple(
            _build_layer(
                stage_id=stage_id,
                layer_id=node_layer_ids[id(node)],
                node=node,
                plan=plan,
                producer_by_tensor=producer_by_tensor,
                node_stage_ids=node_stage_ids,
                tensor_id_by_tensor=tensor_id_by_tensor,
                transition_id_by_stage_layer_input=transition_id_by_stage_layer_input,
            )
            for node in stage_nodes
        ),
    )


def _build_layer(
    stage_id: int,
    layer_id: int,
    node: Node,
    plan: StagePlan,
    producer_by_tensor: dict[object, Node],
    node_stage_ids: dict[int, int],
    tensor_id_by_tensor: dict[object, int],
    transition_id_by_stage_layer_input: dict[tuple[int, int, int], int],
) -> Layer:
    inputs = tuple(
        _build_layer_input(
            stage_id=stage_id,
            layer_id=layer_id,
            input_idx=input_idx,
            tensor=tensor,
            producer=producer_by_tensor.get(tensor),
            node_stage_ids=node_stage_ids,
            tensor_id_by_tensor=tensor_id_by_tensor,
            transition_id_by_stage_layer_input=transition_id_by_stage_layer_input,
        )
        for input_idx, tensor in enumerate(node.inputs)
    )
    outputs = tuple(
        LayerOutput(
            tensor_id=tensor_id_by_tensor[tensor],
            layout=output_layout,
        )
        for tensor, output_layout in zip(node.outputs, _node_output_layout(plan, node))
    )
    return Layer(
        name=node.name,
        layer_id=layer_id,
        kind=node.kind,
        payload=node.payload,
        inputs=inputs,
        outputs=outputs,
    )


def _build_layer_input(
    stage_id: int,
    layer_id: int,
    input_idx: int,
    tensor: object,
    producer: Node | None,
    node_stage_ids: dict[int, int],
    tensor_id_by_tensor: dict[object, int],
    transition_id_by_stage_layer_input: dict[tuple[int, int, int], int],
) -> LayerInput:
    tensor_id = tensor_id_by_tensor[tensor]
    if producer is None:
        return LayerInput.initialization(tensor_id=tensor_id, init_id=tensor_id).source
    if node_stage_ids[id(producer)] == stage_id:
        return LayerInput.stage_input(
            tensor_id=tensor_id,
            layer_id=node_stage_ids[id(producer)],
            output_idx=_node_output_index(producer, tensor),
        ).source
    transition_id = transition_id_by_stage_layer_input[(stage_id, layer_id, input_idx)]
    return LayerInput.transition(tensor_id=tensor_id, transition_id=transition_id).source


def _resolve_stage_selection(
    graph: Graph,
    stage_plans: dict[int, StagePlan],
) -> dict[int, tuple[Node, ...]]:
    if all(plan.nodes for plan in stage_plans.values()):
        return {stage_id: plan.nodes for stage_id, plan in stage_plans.items()}
    return {stage_id: (node,) for stage_id, node in enumerate(graph.nodes)}


def _stage_name(stage_nodes: tuple[Node, ...]) -> str:
    return "+".join(node.name for node in stage_nodes)


def _stage_submesh(plan: StagePlan):
    if plan.node_output_layouts:
        for layouts in plan.node_output_layouts:
            if layouts:
                return layouts[0].submesh
    if plan.output_layouts:
        return plan.output_layouts[0].submesh
    raise ValueError(f"stage {plan.stage_id} has no layouts bound to a submesh")


def _node_output_layout(plan: StagePlan, node: Node):
    if plan.node_output_layouts:
        return plan.node_output_layouts[_plan_node_index(plan, node)]
    return plan.output_layouts


def _plan_node_index(plan: StagePlan, node: Node) -> int:
    for node_idx, candidate in enumerate(plan.nodes):
        if candidate is node:
            return node_idx
    raise ValueError(f"node {node.name} is not present in stage plan {plan.stage_id}")


def _transition_required_slices(
    tensor: object,
    dst_node: Node,
    dst_output_layouts: tuple,
) -> tuple[tuple[Tile, TensorSlice], ...]:
    required_slices = []
    submesh = dst_output_layouts[0].submesh
    for tile in submesh.tiles:
        tile_work = dst_node.payload.build_tile_work(output_layouts=dst_output_layouts, tile=tile)
        for ref in tile_work.input_slices:
            if ref.tensor is tensor:
                required_slices.append((tile, ref.tensor_slice))
                break
    return tuple(required_slices)


def _print_pipeline_stage_cost(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
) -> None:
    stage_selection = _resolve_stage_selection(graph, stage_plans)
    producer_stage_id_by_tensor = _producer_stage_id_by_tensor(stage_selection)
    graph_inputs = frozenset(graph.inputs)
    graph_outputs = frozenset(graph.outputs)
    initializer_tensors = frozenset(graph.initializers)

    worst_stage_compute = 0
    worst_stage_io = 0
    for stage_id, stage_nodes in stage_selection.items():
        plan = stage_plans[stage_id]
        submesh = _stage_submesh(plan)
        worst_stage_compute = max(
            worst_stage_compute,
            _worst_tile_compute_workload_for_stage(
                stage_nodes=stage_nodes,
                node_output_layouts=plan.node_output_layouts,
                submesh=submesh,
            ),
        )
        worst_stage_io = max(
            worst_stage_io,
            _worst_tile_l2_transfer_workload_for_stage(
                stage_id=stage_id,
                stage_nodes=stage_nodes,
                node_output_layouts=plan.node_output_layouts,
                submesh=submesh,
                mesh=mesh,
                graph_inputs=graph_inputs,
                graph_outputs=graph_outputs,
                producer_stage_id_by_tensor=producer_stage_id_by_tensor,
                initializer_tensors=initializer_tensors,
            ),
        )
    print(
        "[planner] pipeline_stage_cost="
        f"{worst_stage_compute + worst_stage_io} "
        f"(worst_compute={worst_stage_compute} worst_io={worst_stage_io})"
    )


def _node_output_index(node: Node, tensor: object) -> int:
    for output_idx, candidate in enumerate(node.outputs):
        if candidate == tensor:
            return output_idx
    raise ValueError(f"tensor {getattr(tensor, 'name', tensor)} is not an output of node {node.name}")
