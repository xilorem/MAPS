"""GEMM-level aggregation built on top of concrete per-tile GEMM work."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.ops.defs.gemm import GemmTileWork


@dataclass(frozen=True)
class GemmCostModel:
    """Compute-only GEMM cycle model backed by tile devices."""

    preferred_device_kind: DeviceKind = DeviceKind.SYSTOLIC

    def cost(self, tile_work: GemmTileWork, tile: Tile) -> int:
        devices = tuple(device for device in tile.devices if device.supports(WorkKind.GEMM))
        preferred = tuple(
            device for device in devices if device.kind is self.preferred_device_kind
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for GEMM work")
        return min(device.cycles(tile_work) for device in candidates)
