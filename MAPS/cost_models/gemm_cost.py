"""GEMM-level aggregation built on top of concrete per-tile GEMM work."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.ops.gemm import GemmTileWork
from MAPS.core.layout import TensorSlice


@dataclass(frozen=True)
class GemmCostModel:
    """Compute-only GEMM cycle model backed by tile devices."""

    preferred_device_kind: DeviceKind = DeviceKind.SYSTOLIC

    def cost(self, tile_work: GemmTileWork, tile: Tile) -> float:
        amount = _gemm_tile_num_ops(tile_work)
        devices = tuple(device for device in tile.devices if device.supports(WorkKind.GEMM))
        preferred = tuple(
            device for device in devices if device.kind is self.preferred_device_kind
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for GEMM work")
        return min(device.cycles(WorkKind.GEMM, amount, tile_work) for device in candidates)


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
