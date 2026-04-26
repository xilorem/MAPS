"""Reduction cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.ops.base import tensor_slice_num_elements
from MAPS.ops.reduction import ReductionTileWork


@dataclass(frozen=True)
class ReductionCostModel:
    """Tile-local reduction cycle model backed by tile devices."""

    work_kind: WorkKind
    preferred_device_kind: DeviceKind = DeviceKind.SCALAR

    def __post_init__(self) -> None:
        if self.work_kind not in (WorkKind.REDUCE_SUM, WorkKind.REDUCE_MAX):
            raise ValueError("ReductionCostModel work_kind must be REDUCE_SUM or REDUCE_MAX")

    def cost(self, tile_work: ReductionTileWork, tile: Tile) -> int:
        amount = tensor_slice_num_elements(tile_work.input_slice)
        devices = tuple(device for device in tile.devices if device.supports(self.work_kind))
        preferred = tuple(
            device for device in devices if device.kind is self.preferred_device_kind
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for {self.work_kind.name} work")
        return min(device.cycles(self.work_kind, amount) for device in candidates)
