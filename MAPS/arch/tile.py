"""Tile-level hardware metadata."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch.device import Device, DeviceKind, WorkKind
from MAPS.arch.memory import L1Memory

DEFAULT_TILE_DEVICES = (
    Device(
        name="core",
        kind=DeviceKind.SCALAR,
        throughput={
            WorkKind.GEMM: 1.0,
            WorkKind.ELEMENTWISE: 1.0,
            WorkKind.REDUCE_SUM: 1.0,
            WorkKind.REDUCE_MAX: 1.0,
            WorkKind.EXP: 1.0,
        },
    ),
)


@dataclass(frozen=True)
class Tile:
    """One physical tile in the mesh."""

    tile_id: int
    x: int
    y: int
    memory: L1Memory = L1Memory(size=1)
    devices: tuple[Device, ...] = DEFAULT_TILE_DEVICES

    def __post_init__(self) -> None:
        if self.tile_id < 0:
            raise ValueError("tile_id must be >= 0")
        if self.x < 0 or self.y < 0:
            raise ValueError("tile coordinates must be >= 0")
        if not self.devices:
            raise ValueError("tile must expose at least one device")

    @property
    def coords(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Tile") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
