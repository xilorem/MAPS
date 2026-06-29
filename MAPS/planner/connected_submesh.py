"""Connected non-rectangular submesh helpers for planner placement."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Mesh, Tile


@dataclass(frozen=True)
class ConnectedSubmesh:
    """One connected placed submesh described by explicit tile ids."""

    mesh: Mesh
    submesh_id: int
    tile_ids: tuple[int, ...]
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.submesh_id < 0:
            raise ValueError("submesh_id must be >= 0")
        if not self.tile_ids:
            raise ValueError("tile_ids must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")
        if self.width * self.height != len(self.tile_ids):
            raise ValueError("logical shape area must match tile count")
        if len(set(self.tile_ids)) != len(self.tile_ids):
            raise ValueError("tile_ids must be unique")
        if any(not self.mesh.contains_tile_id(tile_id) for tile_id in self.tile_ids):
            raise ValueError("tile_ids must be inside the mesh")
        if len(self.tile_ids) > 1 and not _is_connected(self.tile_ids, self.mesh):
            raise ValueError("tile_ids must form one connected component")

    @property
    def num_tiles(self) -> int:
        return len(self.tile_ids)

    @property
    def tiles(self) -> tuple[Tile, ...]:
        return tuple(self.mesh.tile_by_id(tile_id) for tile_id in self.tile_ids)

    @property
    def tile_mask(self) -> int:
        mask = 0
        for tile_id in self.tile_ids:
            mask |= 1 << tile_id
        return mask

    def contains_tile_id(self, tile_id: int) -> bool:
        return tile_id in self.tile_ids

    def intersects_tile_ids(self, tile_ids: set[int]) -> bool:
        return any(tile_id in tile_ids for tile_id in self.tile_ids)

    def global_to_local(self, tile_id: int) -> tuple[int, int]:
        if tile_id not in self.tile_ids:
            raise ValueError(f"tile_id {tile_id} is not inside submesh {self.submesh_id}")
        ordinal = self.tile_ids.index(tile_id)
        return ordinal % self.width, ordinal // self.width

    def local_to_global(self, local_x: int, local_y: int) -> int:
        if local_x < 0 or local_x >= self.width:
            raise ValueError(f"local_x out of bounds: {local_x}")
        if local_y < 0 or local_y >= self.height:
            raise ValueError(f"local_y out of bounds: {local_y}")
        ordinal = local_y * self.width + local_x
        return self.tile_ids[ordinal]


def representative_connected_submesh(
    mesh: Mesh,
    submesh_id: int,
    tile_count: int,
) -> ConnectedSubmesh:
    """Return one deterministic connected submesh used for abstract layout planning."""
    if tile_count <= 0 or tile_count > mesh.num_tiles:
        raise ValueError("tile_count must be in [1, mesh.num_tiles]")
    return ConnectedSubmesh(
        mesh=mesh,
        submesh_id=submesh_id,
        tile_ids=tuple(range(tile_count)),
        width=tile_count,
        height=1,
    )


def connected_submesh_placements(
    tile_count: int,
    mesh: Mesh,
    submesh_id: int,
) -> tuple[ConnectedSubmesh, ...]:
    """Enumerate connected tile placements with one fixed logical line shape."""
    return tuple(
        ConnectedSubmesh(
            mesh=mesh,
            submesh_id=submesh_id,
            tile_ids=tile_ids,
            width=tile_count,
            height=1,
        )
        for tile_ids in _connected_tile_id_sets(tile_count, mesh)
    )


def _connected_tile_id_sets(tile_count: int, mesh: Mesh) -> tuple[tuple[int, ...], ...]:
    if tile_count <= 0:
        raise ValueError("tile_count must be > 0")
    if tile_count > mesh.num_tiles:
        return ()

    neighbors_by_tile_id = {
        tile_id: _cardinal_neighbors(tile_id, mesh)
        for tile_id in range(mesh.num_tiles)
    }
    results: set[tuple[int, ...]] = set()
    seen: set[frozenset[int]] = set()

    def expand(tile_ids: frozenset[int], frontier: frozenset[int]) -> None:
        if tile_ids in seen:
            return
        seen.add(tile_ids)

        if len(tile_ids) == tile_count:
            results.add(tuple(sorted(tile_ids)))
            return

        for next_tile_id in tuple(sorted(frontier)):
            next_tile_ids = tile_ids | {next_tile_id}
            next_frontier = (frontier | neighbors_by_tile_id[next_tile_id]) - next_tile_ids
            expand(next_tile_ids, next_frontier)

    for start_tile_id in range(mesh.num_tiles):
        expand(
            frozenset({start_tile_id}),
            neighbors_by_tile_id[start_tile_id],
        )

    return tuple(sorted(results))


def _cardinal_neighbors(tile_id: int, mesh: Mesh) -> frozenset[int]:
    x, y = mesh.coords(tile_id)
    neighbors = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = x + dx
        ny = y + dy
        if mesh.contains_coord(nx, ny):
            neighbors.add(mesh.tile_id(nx, ny))
    return frozenset(neighbors)


def _is_connected(tile_ids: tuple[int, ...], mesh: Mesh) -> bool:
    remaining = set(tile_ids)
    frontier = {tile_ids[0]}
    visited = set()

    while frontier:
        tile_id = frontier.pop()
        if tile_id in visited:
            continue
        visited.add(tile_id)
        frontier.update(_cardinal_neighbors(tile_id, mesh) & remaining)

    return visited == remaining
