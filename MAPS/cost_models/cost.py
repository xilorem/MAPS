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
