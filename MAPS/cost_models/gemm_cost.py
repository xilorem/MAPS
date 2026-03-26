"""GEMM-level aggregation built on top of concrete per-tile GEMM work."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.builders.gemm_builder import GemmTileWork, build_gemm_tile_work
from MAPS.core.layout import TensorLayout, TensorSlice
from MAPS.ops.gemm import GemmLayerOp


@dataclass(frozen=True)
class GemmCost:
    """Aggregated placeholder cycle estimate of one GEMM for one microbatch."""

    total_ops: int
    tile_work: tuple[GemmTileWork, ...] = field(default_factory=tuple)
    tile_ops: dict[int, int] = field(default_factory=dict)
    tile_costs: dict[int, float] = field(default_factory=dict)
    total_cost: float = 0.0


@dataclass(frozen=True)
class GemmCostModel:
    """Placeholder compute-only GEMM cycle model."""

    ops_per_cycle: float = 1.0
    launch_overhead: float = 0.0

    def __post_init__(self) -> None:
        if self.ops_per_cycle <= 0:
            raise ValueError("ops_per_cycle must be > 0")
        if self.launch_overhead < 0:
            raise ValueError("launch_overhead must be >= 0")

    def cost(self, tile_work: GemmTileWork) -> float:
        """Return a placeholder cycle estimate for one tile."""

        return self.launch_overhead + _gemm_tile_num_ops(tile_work) / self.ops_per_cycle


def _tensor_slice_num_elements(tensor_slice: TensorSlice) -> int:
    total = 1
    for dim in tensor_slice.dims:
        total *= dim.length
    return total


def _gemm_tile_num_ops(tile_work: GemmTileWork) -> int:
    """Return a simple GEMM work count for one tile.

    This is a placeholder compute-only estimate:
    - output elements = owned M x owned N x batch volume
    - reduction depth = full K inferred from the required X slice
    - total ops = output elements x K
    """

    output_elements = _tensor_slice_num_elements(tile_work.output_slice)
    k_depth = tile_work.x_slice.dims[-1].length
    return output_elements * k_depth


def estimate_gemm_cost(op: GemmLayerOp,
                       output_layout: TensorLayout,
                       x_layout: TensorLayout,
                       w_layout: TensorLayout,
                       microbatch_idx: int,
                       model: GemmCostModel) -> GemmCost:
    """Estimate a placeholder GEMM cycle count for one microbatch.

    The current implementation only builds per-tile GEMM work and derives a
    simple compute-only cycle estimate. Transport, overlap, and reduction
    effects are not modeled yet.
    """

    tile_work = tuple(
        build_gemm_tile_work(
            op=op,
            output_layout=output_layout,
            x_layout=x_layout,
            w_layout=w_layout,
            tile=tile,
            microbatch_idx=microbatch_idx,
        )
        for tile in output_layout.submesh.tiles
    )

    tile_ops = {
        tile.tile_id: _gemm_tile_num_ops(work)
        for tile, work in zip(output_layout.submesh.tiles, tile_work)
    }
    tile_costs = {
        tile.tile_id: model.cost(work)
        for tile, work in zip(output_layout.submesh.tiles, tile_work)
    }

    return GemmCost(
        total_ops=sum(tile_ops.values()),
        tile_work=tile_work,
        tile_ops=tile_ops,
        tile_costs=tile_costs,
        total_cost=max(tile_costs.values(), default=0.0),
    )
