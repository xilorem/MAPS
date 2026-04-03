"""GEMM-level aggregation built on top of concrete per-tile GEMM work."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.ops.gemm import GemmTileWork
from MAPS.core.layout import TensorSlice


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
