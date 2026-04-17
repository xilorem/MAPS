from MAPS.arch import CoreDevice, DMADevice, DeviceKind, SystolicDevice
from MAPS.chips.magia import (
    MAGIA_L1_RESERVED_BYTES,
    MAGIA_L1_SIZE_BYTES,
    MAGIA_L1_STACK_BYTES,
    MAGIA_L1_USABLE_BYTES,
    MAGIA_L2_SIZE_BYTES,
    MAGIA_MESH_HEIGHT,
    MAGIA_MESH_WIDTH,
    magia_mesh,
)


def test_magia_mesh_matches_paper_memory_map_and_shape() -> None:
    mesh = magia_mesh()

    assert mesh.shape == (8, 8)
    assert mesh.num_tiles == 64
    assert mesh.l2_memory.size == 1024 * 1024 * 1024
    assert mesh.l2_memory.access_points == tuple((0, y) for y in range(8))
    assert all(tile.memory.size == 1024 * 1024 for tile in mesh.tiles)
    assert MAGIA_MESH_WIDTH == 8
    assert MAGIA_MESH_HEIGHT == 8
    assert MAGIA_L1_SIZE_BYTES == 1024 * 1024
    assert MAGIA_L1_USABLE_BYTES == 896 * 1024
    assert MAGIA_L1_STACK_BYTES == 64 * 1024
    assert MAGIA_L1_RESERVED_BYTES == 64 * 1024
    assert MAGIA_L2_SIZE_BYTES == 1024 * 1024 * 1024


def test_magia_tiles_have_idma_core_and_redmule_devices() -> None:
    mesh = magia_mesh()

    for tile in mesh.tiles:
        devices = {device.name: device for device in tile.devices}
        assert set(devices) == {"idma", "core", "redmule"}
        assert devices["idma"].kind is DeviceKind.DMA
        assert isinstance(devices["idma"], DMADevice)
        assert devices["core"].kind is DeviceKind.SCALAR
        assert isinstance(devices["core"], CoreDevice)
        assert devices["redmule"].kind is DeviceKind.SYSTOLIC
        assert isinstance(devices["redmule"], SystolicDevice)


def test_magia_mesh_accepts_custom_shape() -> None:
    mesh = magia_mesh(width=4, height=3)

    assert mesh.shape == (4, 3)
    assert mesh.num_tiles == 12
    assert mesh.l2_memory.access_points == ((0, 0), (0, 1), (0, 2))
    assert [tile.tile_id for tile in mesh.tiles] == list(range(12))
    assert mesh.tile(3, 2).tile_id == 11
