"""Reusable elementwise cost model."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.ops.base import tensor_slice_num_elements
from MAPS.ops.elementwise import ElementwiseTileWork


@dataclass(frozen=True)
class ElementwiseCostModel:
    """Elementwise cycle model backed by tile devices."""

    work_kind: WorkKind = WorkKind.ELEMENTWISE
    preferred_device_kind: DeviceKind = DeviceKind.SCALAR

    def cost(self, tile_work: ElementwiseTileWork, tile: Tile) -> int:
        amount = tensor_slice_num_elements(tile_work.output_slice)
        devices = tuple(device for device in tile.devices if device.supports(self.work_kind))
        preferred = tuple(
            device for device in devices if device.kind is self.preferred_device_kind
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for {self.work_kind.name} work")
        return min(device.cycles(self.work_kind, amount) for device in candidates)
