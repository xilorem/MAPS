"""Tile-level hardware metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    """One physical tile in the mesh."""

    tile_id: int
    x: int
    y: int
    l1_bytes: int = 1

    @property
    def coords(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Tile") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
