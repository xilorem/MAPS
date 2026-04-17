"""MAGIA chip description."""

from __future__ import annotations

from MAPS.arch import L1Memory, L2Memory, Mesh, Tile
from MAPS.devices import IDMA_DEVICE, SCALAR_CORE_DEVICE
from MAPS.devices.redmule import REDMULE_DEVICE

MAGIA_MESH_WIDTH = 8
MAGIA_MESH_HEIGHT = 8

MAGIA_L1_SIZE_BYTES = 1024 * 1024
MAGIA_L1_USABLE_BYTES = 896 * 1024
MAGIA_L1_STACK_BYTES = 64 * 1024
MAGIA_L1_RESERVED_BYTES = 64 * 1024
MAGIA_L2_SIZE_BYTES = 1024 * 1024 * 1024

MAGIA_L2_ACCESS_POINTS = tuple((0, y) for y in range(MAGIA_MESH_HEIGHT))

MAGIA_IDMA_DEVICE = IDMA_DEVICE
MAGIA_CORE_DEVICE = SCALAR_CORE_DEVICE
MAGIA_REDMULE_DEVICE = REDMULE_DEVICE

MAGIA_TILE_DEVICES = (
    MAGIA_IDMA_DEVICE,
    MAGIA_CORE_DEVICE,
    MAGIA_REDMULE_DEVICE,
)


def magia_mesh(
    width: int = MAGIA_MESH_WIDTH,
    height: int = MAGIA_MESH_HEIGHT,
) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(
            size=MAGIA_L2_SIZE_BYTES,
            access_points=tuple((0, y) for y in range(height)),
        ),
        tiles=tuple(
            Tile(
                tile_id=y * width + x,
                x=x,
                y=y,
                memory=L1Memory(size=MAGIA_L1_USABLE_BYTES),
                devices=MAGIA_TILE_DEVICES,
            )
            for y in range(height)
            for x in range(width)
        ),
    )
