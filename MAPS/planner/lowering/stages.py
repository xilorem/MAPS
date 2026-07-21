"""Lower selected graph nodes into pipeline stages and layers."""

from __future__ import annotations

from MAPS.core.graph import Node
from MAPS.pipeline.layer import Layer, LayerInput, LayerOutput
from MAPS.pipeline.stage import Stage
from MAPS.planner.lowering.context import PipelineLoweringContext
from MAPS.planner.contracts.queries import node_output_layouts
from MAPS.planner.contracts.stages import StagePlacement, StagePlan


def build_stages(
    context: PipelineLoweringContext,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    transition_ids: dict[tuple[int, int, int], int],
) -> tuple[Stage, ...]:
    """Build ordered Pipeline IR stages from selected nodes and planner decisions.

    Node ordering inside each selected stage is preserved.  Inputs produced in
    the same stage become local bindings, graph-boundary inputs become external
    bindings, and cross-stage inputs reference the transitions built earlier.
    Physical placement is attached to each emitted stage without changing its
    virtual tensor layouts.
    """

    return tuple(
        _build_stage(
            stage_id=stage_id,
            plan=stage_plans[stage_id],
            placement=placements[stage_id],
            stage_nodes=stage_nodes,
            context=context,
            transition_ids=transition_ids,
        )
        for stage_id, stage_nodes in context.stage_selection.items()
    )


def _build_stage(
    stage_id: int,
    plan: StagePlan,
    placement: StagePlacement,
    stage_nodes: tuple[Node, ...],
    context: PipelineLoweringContext,
    transition_ids: dict[tuple[int, int, int], int],
) -> Stage:
    """Build one physical stage and its ordered layers."""

    return Stage(
        name="+".join(node.name for node in stage_nodes),
        submesh=placement.physical_submesh,
        virtual_to_physical=placement.virtual_to_physical,
        layers=tuple(
            _build_layer(
                stage_id=stage_id,
                layer_id=context.node_stage_layer_ids[id(node)],
                node=node,
                plan=plan,
                context=context,
                transition_ids=transition_ids,
            )
            for node in stage_nodes
        ),
    )


def _build_layer(
    stage_id: int,
    layer_id: int,
    node: Node,
    plan: StagePlan,
    context: PipelineLoweringContext,
    transition_ids: dict[tuple[int, int, int], int],
) -> Layer:
    """Build one layer and classify each of its tensor bindings."""

    inputs = tuple(
        _build_layer_input(
            stage_id=stage_id,
            layer_id=layer_id,
            input_idx=input_idx,
            tensor=tensor,
            producer=context.producer_by_tensor.get(tensor),
            context=context,
            transition_ids=transition_ids,
        )
        for input_idx, tensor in enumerate(node.inputs)
    )
    outputs = tuple(
        LayerOutput(
            tensor_id=context.tensor_id_by_tensor[tensor],
            layout=output_layout,
        )
        for tensor, output_layout in zip(
            node.outputs,
            node_output_layouts(plan, node),
        )
    )
    return Layer(node=node, inputs=inputs, outputs=outputs)


def _build_layer_input(
    stage_id: int,
    layer_id: int,
    input_idx: int,
    tensor: object,
    producer: Node | None,
    context: PipelineLoweringContext,
    transition_ids: dict[tuple[int, int, int], int],
) -> LayerInput:
    """Classify one layer input as external, local, or transitional."""

    tensor_id = context.tensor_id_by_tensor[tensor]
    if producer is None:
        return LayerInput.external(tensor_id=tensor_id, base_addr=tensor_id + 1)
    if context.node_stage_ids[id(producer)] == stage_id:
        return LayerInput.local(
            tensor_id=tensor_id,
            layer_idx=context.node_stage_layer_ids[id(producer)],
        )
    transition_id = transition_ids[(stage_id, layer_id, input_idx)]
    return LayerInput.transition(tensor_id=tensor_id, transition_id=transition_id)
