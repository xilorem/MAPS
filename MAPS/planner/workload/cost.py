"""Generic planner-side node cost estimation entry points."""

from __future__ import annotations

from MAPS.core.graph import Node
from MAPS.core.layout import TensorLayout


def cost_estimator(
    node: Node,
    output_layouts: tuple[TensorLayout, ...],
) -> int:
    """Estimate one node's bottleneck compute cost for virtual planning."""

    cost_model = node.payload.cost_model
    placement_cost = getattr(cost_model, "placement_cost", None)
    if placement_cost is not None:
        return int(placement_cost(node=node, output_layouts=output_layouts))
    submesh = output_layouts[0].submesh
    tile_work = tuple(
        (
            tile,
            node.payload.build_tile_work(output_layouts=output_layouts, tile=tile),
        )
        for tile in submesh.tiles
    )
    return max(
        (cost_model.cost(work, tile) for tile, work in tile_work),
        default=0,
    )


def placement_cost_estimator(
    node: Node,
    output_layouts: tuple[TensorLayout, ...],
) -> int:
    """Estimate the placement-specific component of one node cost."""

    placement_cost = getattr(node.payload.cost_model, "placement_cost", None)
    if placement_cost is None:
        return 0
    return int(placement_cost(node=node, output_layouts=output_layouts))
