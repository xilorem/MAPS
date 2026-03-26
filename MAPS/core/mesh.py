"""
Small geometry helpers for mesh-level planning.

Examples
--------
Create a 4x4 mesh and inspect one tile:

    >>> mesh = Mesh(4, 4)
    >>> tile = mesh.tile(2, 1)
    >>> tile
    Tile(tile_id=6, x=2, y=1)
    >>> mesh.coords(6)
    (2, 1)

Extract a rectangular placed submesh and measure distances:

    >>> mesh = Mesh(4, 4)
    >>> rect = mesh.rectangle(x0=1, y0=1, width=2, height=2)
    >>> [tile.tile_id for tile in rect]
    [5, 6, 9, 10]
    >>> mesh.manhattan_distance(mesh.tile(0, 0), mesh.tile(3, 2))
    5

Enumerate every rectangular placement on a small mesh:

    >>> mesh = Mesh(2, 2)
    >>> rectangles = mesh.all_rectangles()
    >>> len(rectangles)
    9
    >>> [tile.tile_id for tile in rectangles[0]]
    [0]
    >>> [tile.tile_id for tile in rectangles[-1]]
    [0, 1, 2, 3]
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tile:
    """
    One physical tile in the mesh.

    Example
    -------
        >>> mesh = Mesh(4, 4)
        >>> mesh.tile_by_id(3)
        Tile(tile_id=3, x=3, y=0)
    """

    tile_id: int
    x: int
    y: int

    def manhattan_distance(self, other: "Tile") -> int:
        """Return Manhattan distance to another tile."""

        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class Mesh:
    """
    Rectangular mesh of tiles stored in row-major order.

    Tile IDs follow the same convention used by the runtime side of the repo:
    `tile_id = y * x_size + x`.

    Example
    -------
        >>> mesh = Mesh(3, 2)
        >>> [tile.tile_id for tile in mesh.tiles]
        [0, 1, 2, 3, 4, 5]
        >>> [tile.tile_id for tile in mesh.row(1)]
        [3, 4, 5]
    """

    x_size: int
    y_size: int
    _tiles: tuple[Tile, ...] = field(init=False, repr=False)

    # create tiles inside the mesh
    def __post_init__(self) -> None:
        if self.x_size <= 0:
            raise ValueError("x_size must be > 0")
        if self.y_size <= 0:
            raise ValueError("y_size must be > 0")

        tiles = []
        for y in range(self.y_size):
            for x in range(self.x_size):
                tiles.append(Tile(tile_id=self.tile_id(x, y), x=x, y=y))

        self._tiles = tuple(tiles)

    @property
    def num_tiles(self) -> int:
        return self.x_size * self.y_size

    @property
    def tiles(self) -> tuple[Tile, ...]:
        return self._tiles

    def contains_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.x_size and 0 <= y < self.y_size

    def contains_tile_id(self, tile_id: int) -> bool:
        return 0 <= tile_id < self.num_tiles

    def tile_id(self, x: int, y: int) -> int:
        if not self.contains_coord(x, y):
            raise ValueError(f"coordinates out of bounds: ({x}, {y})")
        return y * self.x_size + x

    def coords(self, tile_id: int) -> tuple[int, int]:
        if not self.contains_tile_id(tile_id):
            raise ValueError(f"tile_id out of bounds: {tile_id}")
        return tile_id % self.x_size, tile_id // self.x_size

    def tile(self, x: int, y: int) -> Tile:
        return self.tile_by_id(self.tile_id(x, y))

    def tile_by_id(self, tile_id: int) -> Tile:
        if not self.contains_tile_id(tile_id):
            raise ValueError(f"tile_id out of bounds: {tile_id}")
        return self._tiles[tile_id]

    def row(self, y: int) -> tuple[Tile, ...]:
        if y < 0 or y >= self.y_size:
            raise ValueError(f"row out of bounds: {y}")
        start = y * self.x_size
        end = start + self.x_size
        return self._tiles[start:end]

    def column(self, x: int) -> tuple[Tile, ...]:
        if x < 0 or x >= self.x_size:
            raise ValueError(f"column out of bounds: {x}")
        return tuple(self.tile(x, y) for y in range(self.y_size))

    def rectangle(self,
                  x0: int,
                  y0: int,
                  width: int,
                  height: int) -> tuple[Tile, ...]:
        """
        Return the tiles in one rectangular placed submesh in row-major order.
        """

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if not self.contains_coord(x0, y0):
            raise ValueError(f"rectangle origin out of bounds: ({x0}, {y0})")
        if not self.contains_coord(x0 + width - 1, y0 + height - 1):
            raise ValueError("rectangle exceeds mesh bounds")

        return tuple(
            self.tile(x, y)
            for y in range(y0, y0 + height)
            for x in range(x0, x0 + width)
        )

    def all_rectangles(self) -> tuple[tuple[Tile, ...], ...]:
        """
        Enumerate every rectangular submesh placement on this mesh.

        This is a good primitive for early-stage planner search.
        """

        rectangles = []
        for height in range(1, self.y_size + 1):
            for width in range(1, self.x_size + 1):
                for y0 in range(0, self.y_size - height + 1):
                    for x0 in range(0, self.x_size - width + 1):
                        rectangles.append(self.rectangle(x0, y0, width, height))

        return tuple(rectangles)

    def manhattan_distance(self, a: Tile, b: Tile) -> int:
        return a.manhattan_distance(b)
