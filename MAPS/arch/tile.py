"""Tile-level hardware metadata."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch.device import Device
from MAPS.arch.memory import L1Memory
from MAPS.devices.generic import GENERIC_CORE_DEVICE

DEFAULT_TILE_DEVICES = (GENERIC_CORE_DEVICE,)


@dataclass(frozen=True)
class Tile:
    """One physical tile in the mesh."""

    tile_id: int
    x: int
    y: int
    devices: tuple[Device, ...] = DEFAULT_TILE_DEVICES
    memory: L1Memory = L1Memory(size=1)

    def __post_init__(self) -> None:
        if self.tile_id < 0:
            raise ValueError("tile_id must be >= 0")
        if self.x < 0 or self.y < 0:
            raise ValueError("tile coordinates must be >= 0")
        if not self.devices:
            object.__setattr__(self, "devices", DEFAULT_TILE_DEVICES)

    @property
    def coords(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Tile") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
