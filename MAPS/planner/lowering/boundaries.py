"""Lower graph inputs and outputs into pipeline boundary transfers."""

from __future__ import annotations

from MAPS.core.layout import tile_tensor_slice
from MAPS.pipeline.finalization import Finalization, FinalizationFragment
from MAPS.pipeline.initialization import Initialization, InitializationFragment
from MAPS.planner.lowering.context import PipelineLoweringContext
from MAPS.planner.contracts.queries import (
    node_output_index,
    node_output_layouts,
    required_input_slices,
)
from MAPS.planner.contracts.stages import StagePlacement, StagePlan


def build_initializations(
    context: PipelineLoweringContext,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
) -> tuple[Initialization, ...]:
    """Build transfers that load graph-boundary inputs onto consumer tiles.

    Every node input without an in-graph producer is an external input.  Its
    required slice is derived from the consumer's tile work, and one fragment is
    emitted per destination tile using the final physical placement.
    """

    initializations = []
    for stage_id, stage_nodes in context.stage_selection.items():
        plan = stage_plans[stage_id]
        placement = placements[stage_id]
        for node in stage_nodes:
            output_layouts = node_output_layouts(plan, node)
            for input_idx, tensor in enumerate(node.inputs):
                if tensor in context.producer_by_tensor:
                    continue
                required_slices = required_input_slices(
                    tensor=tensor,
                    destination_node=node,
                    destination_output_layouts=output_layouts,
                )
                initializations.append(
                    Initialization(
                        name=f"init_{tensor.name}",
                        tensor_id=context.tensor_id_by_tensor[tensor],
                        dst_layer_id=context.node_graph_layer_ids[id(node)],
                        dst_input_idx=input_idx,
                        fragments=tuple(
                            InitializationFragment(
                                src_hartid=-1,
                                dst_hartid=placement.physical_tile_id(tile.tile_id),
                                src_slice=tensor_slice,
                                dst_slice=tensor_slice,
                            )
                            for tile, tensor_slice in required_slices
                        ),
                    )
                )
    return tuple(initializations)


def build_finalizations(
    context: PipelineLoweringContext,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
) -> tuple[Finalization, ...]:
    """Build transfers that copy graph outputs from producer tiles to the host.

    Finalization fragments cover each virtual output-layout slice exactly once
    and translate its producer tile through the selected physical placement.
    """

    finalizations = []
    for tensor in context.graph.outputs:
        source_node = context.producer_by_tensor[tensor]
        source_stage_id = context.node_stage_ids[id(source_node)]
        source_plan = stage_plans[source_stage_id]
        source_placement = placements[source_stage_id]
        source_output_idx = node_output_index(source_node, tensor)
        source_layout = node_output_layouts(source_plan, source_node)[source_output_idx]
        finalizations.append(
            Finalization(
                name=f"output_{tensor.name}",
                tensor_id=context.tensor_id_by_tensor[tensor],
                src_layer_id=context.node_graph_layer_ids[id(source_node)],
                src_output_idx=source_output_idx,
                fragments=tuple(
                    FinalizationFragment(
                        src_hartid=source_placement.physical_tile_id(tile.tile_id),
                        dst_hartid=-1,
                        src_slice=tile_tensor_slice(tensor, source_layout, tile),
                        dst_slice=tile_tensor_slice(tensor, source_layout, tile),
                    )
                    for tile in source_layout.submesh.tiles
                ),
            )
        )
    return tuple(finalizations)
