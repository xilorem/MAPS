"""Placed submesh IR matching the runtime-side `submesh_t`."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Mesh, Tile


@dataclass(frozen=True)
class Submesh:
    """One rectangular placed submesh inside a mesh."""

    mesh: Mesh
    submesh_id: int
    x0: int
    y0: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.submesh_id < 0:
            raise ValueError("submesh_id must be >= 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")
        if not self.mesh.contains_coord(self.x0, self.y0):
            raise ValueError(f"submesh origin out of bounds: ({self.x0}, {self.y0})")
        if not self.mesh.contains_coord(self.x0 + self.width - 1, self.y0 + self.height - 1):
            raise ValueError("submesh exceeds mesh bounds")

    @property
    def num_tiles(self) -> int:
        return self.width * self.height

    @property
    def tiles(self) -> tuple[Tile, ...]:
        return self.mesh.rectangle(self.x0, self.y0, self.width, self.height)

    def contains_tile_id(self, tile_id: int) -> bool:
        if not self.mesh.contains_tile_id(tile_id):
            return False
        x, y = self.mesh.coords(tile_id)
        return self.x0 <= x < self.x0 + self.width and self.y0 <= y < self.y0 + self.height

    def local_coords(self, tile_id: int) -> tuple[int, int]:
        if not self.contains_tile_id(tile_id):
            raise ValueError(f"tile_id {tile_id} is not inside submesh {self.submesh_id}")
        x, y = self.mesh.coords(tile_id)
        return x - self.x0, y - self.y0

    def tile_id_at(self, local_x: int, local_y: int) -> int:
        if local_x < 0 or local_x >= self.width:
            raise ValueError(f"local_x out of bounds: {local_x}")
        if local_y < 0 or local_y >= self.height:
            raise ValueError(f"local_y out of bounds: {local_y}")
        return self.mesh.tile_id(self.x0 + local_x, self.y0 + local_y)
