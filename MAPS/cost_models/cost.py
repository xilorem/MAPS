"""Generic node-based cost estimation entry points."""

from __future__ import annotations

from MAPS.core.graph import Node
from MAPS.core.layout import TensorLayout


def cost_estimator(
    node: Node,
    input_layouts: tuple[TensorLayout, ...],
    output_layouts: tuple[TensorLayout, ...],
) -> float:
    """Estimate one node per-step cost from explicit execution assumptions."""

    cost_model = node.payload.cost_model
    placement_cost = getattr(cost_model, "placement_cost", None)
    if placement_cost is not None:
        return float(
            placement_cost(
                node=node,
                input_layouts=input_layouts,
                output_layouts=output_layouts,
            )
        )
    submesh = output_layouts[0].submesh
    tile_work = tuple(
        (
            tile,
            node.payload.build_tile_work(
                input_layouts=input_layouts,
                output_layouts=output_layouts,
                tile=tile,
            ),
        )
        for tile in submesh.tiles
    )
    return max((cost_model.cost(work, tile) for tile, work in tile_work), default=0.0)


def placement_cost_estimator(
    node: Node,
    input_layouts: tuple[TensorLayout, ...],
    output_layouts: tuple[TensorLayout, ...],
) -> float:
    """Estimate one node cost for one concrete placement.

    Cost models can optionally override this with ``placement_cost(...)`` when
    their latency depends on the concrete submesh placement rather than only on
    per-tile work. Otherwise this falls back to the existing node cost.
    """

    cost_model = node.payload.cost_model
    placement_cost = getattr(cost_model, "placement_cost", None)
    if placement_cost is None:
        return cost_estimator(node=node, input_layouts=input_layouts, output_layouts=output_layouts)
    return float(
        placement_cost(
            node=node,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
        )
    )
