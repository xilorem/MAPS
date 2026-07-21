"""Virtual communication analysis performed before physical placement."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.core.layout import tile_tensor_slice
from MAPS.planner.contracts.queries import (
    node_output_index,
    node_output_layouts,
    required_input_slices,
)
from MAPS.planner.contracts.stages import StagePlan, virtual_submesh
from MAPS.planner.spatial.models import VirtualTraffic
from MAPS.transitions import build_direct_remap_fragments


def build_virtual_traffic(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    node_stage_ids: dict[int, int],
) -> VirtualTraffic:
    """Describe all stage communication before physical tiles are selected.

    Contract:
        Stage plans must contain final virtual layouts, and ``node_stage_ids``
        must identify every planned node.  Physical placement is deliberately
        absent from this analysis.

    Returns:
        Per-edge virtual-tile byte matrices plus aggregate input, output, L2,
        communication-degree, and bottleneck-pressure weights.  These values are
        byte counts, not transport cycles.
    """

    del mesh
    producer_by_tensor = {
        tensor: node
        for node in graph.nodes
        for tensor in node.outputs
    }
    initializer_tensors = frozenset(graph.initializers)
    graph_inputs = frozenset(graph.inputs) - initializer_tensors
    graph_outputs = frozenset(graph.outputs)
    stage_ids = tuple(stage_plans)

    stage_comm: dict[tuple[int, int], int] = {}
    edge_matrices: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    input_weights = _empty_stage_tile_weights(stage_plans)
    output_weights = _empty_stage_tile_weights(stage_plans)
    l2_read_weights = _empty_stage_tile_weights(stage_plans)
    l2_write_weights = _empty_stage_tile_weights(stage_plans)

    for destination_stage_id, destination_plan in stage_plans.items():
        for destination_node, output_layouts in zip(
            destination_plan.nodes,
            destination_plan.node_output_layouts,
        ):
            for tensor in destination_node.inputs:
                if tensor in initializer_tensors:
                    continue
                required_slices = required_input_slices(
                    tensor=tensor,
                    destination_node=destination_node,
                    destination_output_layouts=output_layouts,
                )
                source_node = producer_by_tensor.get(tensor)
                if tensor in graph_inputs or source_node is None:
                    for destination_tile, tensor_slice in required_slices:
                        bytes_ = tensor.slice_num_bytes(tensor_slice)
                        tile_id = destination_tile.tile_id
                        input_weights[destination_stage_id][tile_id] += bytes_
                        l2_read_weights[destination_stage_id][tile_id] += bytes_
                    continue

                source_stage_id = node_stage_ids[id(source_node)]
                if source_stage_id == destination_stage_id:
                    continue
                source_layout = node_output_layouts(
                    stage_plans[source_stage_id],
                    source_node,
                )[node_output_index(source_node, tensor)]
                fragments = build_direct_remap_fragments(
                    tensor=tensor,
                    src_layout=source_layout,
                    dst_required_slices=required_slices,
                )
                matrix = edge_matrices.setdefault(
                    (source_stage_id, destination_stage_id),
                    {},
                )
                for fragment in fragments:
                    bytes_ = fragment.src_subslice.num_elements * tensor.elem_bytes
                    key = (fragment.src_hartid, fragment.dst_hartid)
                    matrix[key] = matrix.get(key, 0) + bytes_
                    edge = (source_stage_id, destination_stage_id)
                    stage_comm[edge] = stage_comm.get(edge, 0) + bytes_
                    output_weights[source_stage_id][fragment.src_hartid] += bytes_
                    input_weights[destination_stage_id][fragment.dst_hartid] += bytes_

            for output_idx, tensor in enumerate(destination_node.outputs):
                if tensor not in graph_outputs:
                    continue
                output_layout = output_layouts[output_idx]
                for virtual_tile in output_layout.submesh.tiles:
                    bytes_ = tensor.slice_num_bytes(
                        tile_tensor_slice(tensor, output_layout, virtual_tile)
                    )
                    tile_id = virtual_tile.tile_id
                    output_weights[destination_stage_id][tile_id] += bytes_
                    l2_write_weights[destination_stage_id][tile_id] += bytes_

    communication_degree = {
        stage_id: sum(
            weight
            for (source_stage_id, destination_stage_id), weight in stage_comm.items()
            if source_stage_id == stage_id or destination_stage_id == stage_id
        )
        for stage_id in stage_ids
    }
    bottleneck_risk = {
        stage_id: max(input_weights[stage_id].values(), default=0)
        for stage_id in stage_ids
    }
    l2_pressure = {
        stage_id: (
            sum(l2_read_weights[stage_id].values())
            + sum(l2_write_weights[stage_id].values())
        )
        for stage_id in stage_ids
    }
    return VirtualTraffic(
        stage_comm=stage_comm,
        edge_matrices=edge_matrices,
        input_weights=input_weights,
        output_weights=output_weights,
        l2_read_weights=l2_read_weights,
        l2_write_weights=l2_write_weights,
        communication_degree=communication_degree,
        bottleneck_risk=bottleneck_risk,
        l2_pressure=l2_pressure,
    )


def _empty_stage_tile_weights(
    stage_plans: dict[int, StagePlan],
) -> dict[int, dict[int, int]]:
    """Build zero-valued virtual-tile weights for every stage."""

    return {
        stage_id: {
            tile.tile_id: 0
            for tile in virtual_submesh(plan).tiles
        }
        for stage_id, plan in stage_plans.items()
    }
