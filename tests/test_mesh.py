from MAPS.arch import Mesh, Tile


def test_mesh_preserves_legacy_dimension_accessors() -> None:
    mesh = Mesh(width=3, height=2, l2_bytes=8192)

    assert mesh.width == 3
    assert mesh.height == 2
    assert mesh.x_size == 3
    assert mesh.y_size == 2
    assert mesh.shape == (3, 2)
    assert mesh.num_tiles == 6
    assert mesh.l2_bytes == 8192


def test_mesh_can_use_explicit_tiles_with_l1_capacity() -> None:
    tiles = (
        Tile(tile_id=0, x=0, y=0, l1_bytes=4096),
        Tile(tile_id=1, x=1, y=0, l1_bytes=4096),
        Tile(tile_id=2, x=0, y=1, l1_bytes=2048),
        Tile(tile_id=3, x=1, y=1, l1_bytes=2048),
    )
    mesh = Mesh(width=2, height=2, l2_bytes=16384, tiles=tiles)

    assert tuple(tile.l1_bytes for tile in mesh.tiles) == (4096, 4096, 2048, 2048)
    assert mesh.tile(1, 1).l1_bytes == 2048


def test_mesh_rectangle_keeps_row_major_order() -> None:
    mesh = Mesh(width=4, height=3, l2_bytes=4096)

    rectangle = mesh.rectangle(x0=1, y0=1, width=2, height=2)

    assert [tile.tile_id for tile in rectangle] == [5, 6, 9, 10]
