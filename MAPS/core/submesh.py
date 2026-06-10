"""Placed submesh IR matching the runtime-side `submesh_t`."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Mesh, Tile


def _adjacent_tile_ids(mesh: Mesh, tile_id: int) -> set[int]:
    tile = mesh.tile_by_id(tile_id)
    neighbors: set[int] = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x = tile.x + dx
        y = tile.y + dy
        if 0 <= x < mesh.width and 0 <= y < mesh.height:
            neighbors.add(mesh.tile_id(x, y))
    return neighbors


def _is_connected_tile_set(mesh: Mesh, tile_ids: frozenset[int]) -> bool:
    if not tile_ids:
        return False
    start = next(iter(tile_ids))
    seen = {start}
    stack = [start]
    while stack:
        tile_id = stack.pop()
        for neighbor in _adjacent_tile_ids(mesh, tile_id):
            if neighbor in tile_ids and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(tile_ids)


@dataclass(frozen=True)
class Submesh:
    """One connected placed submesh inside a mesh.

    The shape does not need to be rectangular, but it must be 4-neighbor connected.
    """

    mesh: Mesh
    submesh_id: int
    tile_ids: frozenset[int] | set[int]

    def __post_init__(self) -> None:
        if self.submesh_id < 0:
            raise ValueError("submesh_id must be >= 0")

        tile_ids = frozenset(self.tile_ids)
        object.__setattr__(self, "tile_ids", tile_ids)

        if not tile_ids:
            raise ValueError("tile_ids must not be empty")

        if len(tile_ids) > 1:
            isolated = [
                tile_id
                for tile_id in tile_ids
                if not (_adjacent_tile_ids(self.mesh, tile_id) & tile_ids)
            ]
            if isolated:
                raise ValueError(
                    f"submesh contains isolated tiles: {isolated}"
                )
            if not _is_connected_tile_set(self.mesh, tile_ids):
                raise ValueError(
                    f"submesh tiles must form one connected component: {set(tile_ids)}"
                )

    @property
    def num_tiles(self) -> int:
        """Return the number of tiles covered by this submesh."""
        return len(self.tile_ids)

    @property
    def tiles(self) -> tuple[Tile, ...]:
        """Return mesh tiles covered by this submesh in row-major order."""
        return tuple(
            self.mesh.tile_by_id(tile_id)
            for tile_id in sorted(
                self.tile_ids,
                key=lambda tile_id: (
                    self.mesh.tile_by_id(tile_id).y,
                    self.mesh.tile_by_id(tile_id).x,
                ),
            )
        )

    @property
    def tile_mask(self) -> int:
        """Return a bit mask for fast overlap checks."""
        mask = 0
        for tile_id in self.tile_ids:
            mask |= 1 << tile_id
        return mask

    @property
    def x0(self) -> int:
        """Left edge of the bounding box. Compatibility with old rectangular code."""
        return min(tile.x for tile in self.tiles)

    @property
    def y0(self) -> int:
        """Top edge of the bounding box. Compatibility with old rectangular code."""
        return min(tile.y for tile in self.tiles)

    @property
    def width(self) -> int:
        """Bounding-box width. Compatibility with old rectangular code."""
        xs = [tile.x for tile in self.tiles]
        return max(xs) - min(xs) + 1

    @property
    def height(self) -> int:
        """Bounding-box height. Compatibility with old rectangular code."""
        ys = [tile.y for tile in self.tiles]
        return max(ys) - min(ys) + 1

    def intersects_tile_ids(self, tile_ids: set[int]) -> bool:
        """Return whether this submesh overlaps any global mesh tile id in the set."""
        return bool(self.tile_ids & tile_ids)