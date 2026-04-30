"""GEMM-level aggregation built on top of concrete per-tile GEMM work."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.ops.defs.gemm import GemmTileWork


@dataclass(frozen=True)
class GemmCostModel:
    """Compute-only GEMM cycle model backed by tile devices."""

    preferred_device_kinds: DeviceKind | tuple[DeviceKind, ...] = (DeviceKind.MATRIX, DeviceKind.SYSTOLIC)

    def cost(self, tile_work: GemmTileWork, tile: Tile) -> int:
        devices = tuple(device for device in tile.devices if device.supports(WorkKind.GEMM))
        preferred_kinds = self._preferred_device_kinds()
        preferred = tuple(
            device for device in devices if device.kind in preferred_kinds
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for GEMM work")
        return min(device.cycles(tile_work) for device in candidates)

    def _preferred_device_kinds(self) -> tuple[DeviceKind, ...]:
        preferred = self.preferred_device_kinds
        if isinstance(preferred, DeviceKind):
            return (preferred,)
        return tuple(preferred)
