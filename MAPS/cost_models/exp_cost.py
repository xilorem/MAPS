"""Exp cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.core.layout import TensorSlice
from MAPS.ops.exp import ExpTileWork


@dataclass(frozen=True)
class ExpCostModel:
    """Elementwise exponential cycle model backed by tile devices."""

    preferred_device_kind: DeviceKind = DeviceKind.SCALAR

    def cost(self, tile_work: ExpTileWork, tile: Tile) -> float:
        amount = _tensor_slice_num_elements(tile_work.output_slice)
        devices = tuple(device for device in tile.devices if device.supports(WorkKind.EXP))
        preferred = tuple(
            device for device in devices if device.kind is self.preferred_device_kind
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for Exp work")
        return min(device.cycles(WorkKind.EXP, amount, tile_work) for device in candidates)


def _tensor_slice_num_elements(tensor_slice: TensorSlice) -> int:
    total = 1
    for dim in tensor_slice.dims:
        total *= dim.length
    return total
