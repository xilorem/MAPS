"""Mesh-level geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory import L2Memory
from .noc import NoC
from .tile import Tile


@dataclass(frozen=True)
class Mesh:
    """Rectangular mesh of tiles stored in row-major order."""

    width: int
    height: int
    l2_memory: L2Memory
    noc: NoC | None
    _tiles: tuple[Tile, ...] = field(init=False, repr=False)
    
    def __init__(
        self,
        width: int,
        height: int,      
        l2_memory: L2Memory = L2Memory(size=1),
        tiles: tuple[Tile, ...] | None = None,
        noc: NoC | None = None,
    ) -> None:
        if width <= 0:
            raise ValueError("width must be > 0")
        if height <= 0:
            raise ValueError("height must be > 0")

        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "l2_memory", l2_memory)
        if tiles is None:
            tiles = tuple(
                Tile(tile_id=(y * width + x), x=x, y=y)
                for y in range(height)
                for x in range(width)
            )
        else:
            self._validate_tiles(width, height, tiles)

        self._validate_noc(width, height, noc)
        object.__setattr__(self, "noc", noc)
        object.__setattr__(self, "_tiles", tiles)

    @staticmethod
    def _validate_tiles(width: int, height: int, tiles: tuple[Tile, ...]) -> None:
        if len(tiles) != width * height:
            raise ValueError("tiles length must match mesh area")

        for expected_id, tile in enumerate(tiles):
            if tile.tile_id != expected_id:
                raise ValueError("tiles must be ordered by row-major tile_id")
            expected_x = expected_id % width
            expected_y = expected_id // width
            if tile.x != expected_x or tile.y != expected_y:
                raise ValueError("tile coordinates must match row-major tile placement")

    @staticmethod
    def _validate_noc(width: int, height: int, noc: NoC | None) -> None:
        if noc is None:
            return

        num_tiles = width * height
        for endpoint in noc.endpoints:
            if endpoint.tile_id is None:
                continue
            if not (0 <= endpoint.tile_id < num_tiles):
                raise ValueError(f"NoC endpoint tile_id out of bounds: {endpoint.tile_id}")

    @property
    def x_size(self) -> int:
        return self.width

    @property
    def y_size(self) -> int:
        return self.height

    @property
    def shape(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def num_tiles(self) -> int:
        return self.width * self.height

    @property
    def tiles(self) -> tuple[Tile, ...]:
        return self._tiles

    @property
    def has_noc(self) -> bool:
        return self.noc is not None

    def contains_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def contains_tile_id(self, tile_id: int) -> bool:
        return 0 <= tile_id < self.num_tiles

    def tile_id(self, x: int, y: int) -> int:
        if not self.contains_coord(x, y):
            raise ValueError(f"coordinates out of bounds: ({x}, {y})")
        return y * self.width + x

    def coords(self, tile_id: int) -> tuple[int, int]:
        if not self.contains_tile_id(tile_id):
            raise ValueError(f"tile_id out of bounds: {tile_id}")
        return tile_id % self.width, tile_id // self.width

    def tile(self, x: int, y: int) -> Tile:
        return self.tile_by_id(self.tile_id(x, y))

    def tile_by_id(self, tile_id: int) -> Tile:
        if not self.contains_tile_id(tile_id):
            raise ValueError(f"tile_id out of bounds: {tile_id}")
        return self._tiles[tile_id]

    def row(self, y: int) -> tuple[Tile, ...]:
        if y < 0 or y >= self.height:
            raise ValueError(f"row out of bounds: {y}")
        start = y * self.width
        return self._tiles[start:start + self.width]

    def column(self, x: int) -> tuple[Tile, ...]:
        if x < 0 or x >= self.width:
            raise ValueError(f"column out of bounds: {x}")
        return tuple(self.tile(x, y) for y in range(self.height))

    def rectangle(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> tuple[Tile, ...]:
        """Return one rectangular placed region in row-major tile order."""

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
        """Enumerate every rectangular placement on this mesh."""

        rectangles = []
        for height in range(1, self.height + 1):
            for width in range(1, self.width + 1):
                for y0 in range(0, self.height - height + 1):
                    for x0 in range(0, self.width - width + 1):
                        rectangles.append(self.rectangle(x0, y0, width, height))
        return tuple(rectangles)

    def manhattan_distance(self, a: Tile, b: Tile) -> int:
        return a.manhattan_distance(b)
