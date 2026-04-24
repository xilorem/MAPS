"""Tile-level hardware metadata."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch.device import Device
from MAPS.arch.memory import L1Memory


@dataclass(frozen=True)
class Tile:
    """One physical tile in the mesh."""

    tile_id: int
    x: int
    y: int
    memory: L1Memory
    devices: tuple[Device, ...]

    def __post_init__(self) -> None:
        # check for valid tile identification and mesh position
        if self.tile_id < 0:
            raise ValueError("tile_id must be >= 0")
        if self.x < 0 or self.y < 0:
            raise ValueError("tile coordinates must be >= 0")
        if not self.devices:
            raise ValueError("tile devices must not be empty")
