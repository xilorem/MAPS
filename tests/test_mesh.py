import pytest

from MAPS.arch import EndpointKind, L1Memory, L2Memory, Mesh, NoC, NoCChannel, NoCEndpoint, NoCLink, NoCNode, Tile


def test_mesh_preserves_dimension_accessors() -> None:
    mesh = Mesh(width=3, height=2, l2_memory=L2Memory(size=8192))

    assert mesh.width == 3
    assert mesh.height == 2
    assert mesh.x_size == 3
    assert mesh.y_size == 2
    assert mesh.shape == (3, 2)
    assert mesh.num_tiles == 6
    assert mesh.l2_memory.size == 8192


def test_mesh_can_use_explicit_tiles_with_l1_capacity() -> None:
    tiles = (
        Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096)),
        Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096)),
        Tile(tile_id=2, x=0, y=1, memory=L1Memory(size=2048)),
        Tile(tile_id=3, x=1, y=1, memory=L1Memory(size=2048)),
    )
    mesh = Mesh(width=2, height=2, l2_memory=L2Memory(size=16384), tiles=tiles)

    assert tuple(tile.memory.size for tile in mesh.tiles) == (4096, 4096, 2048, 2048)
    assert mesh.tile(1, 1).memory.size == 2048


def test_mesh_exposes_memory_objects() -> None:
    tiles = (
        Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=64)),
        Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=2048, bandwidth=32)),
    )
    mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=16384, access_points=((0, 0),), bandwidth=128),
        tiles=tiles,
    )

    assert mesh.l2_memory.size == 16384
    assert mesh.l2_memory.access_points == ((0, 0),)
    assert mesh.l2_memory.bandwidth == 128
    assert mesh.tile(0, 0).memory.bandwidth == 64
    assert tuple(tile.memory.size for tile in mesh.tiles) == (4096, 2048)


def test_mesh_rejects_l2_access_points_outside_mesh() -> None:
    with pytest.raises(ValueError, match="L2 access point out of bounds"):
        Mesh(
            width=2,
            height=2,
            l2_memory=L2Memory(size=4096, access_points=((2, 0),)),
        )


def test_mesh_rectangle_keeps_row_major_order() -> None:
    mesh = Mesh(width=4, height=3, l2_memory=L2Memory(size=4096))

    rectangle = mesh.rectangle(x0=1, y0=1, width=2, height=2)

    assert [tile.tile_id for tile in rectangle] == [5, 6, 9, 10]


def test_mesh_accepts_attached_noc() -> None:
    noc = NoC(
        nodes=(
            NoCNode(node_id=0, x=0, y=0),
            NoCNode(node_id=1, x=1, y=0),
        ),
        links=(
            NoCLink(
                link_id=0,
                src_node_id=0,
                dst_node_id=1,
                channels=(NoCChannel(channel_id=0, width_bytes=4),),
                bidirectional=True,
            ),
        ),
        endpoints=(
            NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
            NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=1),
        ),
    )

    mesh = Mesh(width=2, height=1, l2_memory=L2Memory(size=4096), noc=noc)

    assert mesh.noc == noc
    assert mesh.has_noc is True
    assert mesh.noc.endpoint_by_id(0).tile_id == 0


def test_mesh_defaults_to_no_noc() -> None:
    mesh = Mesh(width=2, height=1, l2_memory=L2Memory(size=4096))

    assert mesh.noc is None
    assert mesh.has_noc is False


def test_mesh_rejects_attached_noc_endpoint_tile_id_outside_mesh() -> None:
    noc = NoC(
        nodes=(NoCNode(node_id=0, x=0, y=0),),
        links=(),
        endpoints=(NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=4),),
    )

    with pytest.raises(ValueError, match="NoC endpoint tile_id out of bounds"):
        Mesh(width=2, height=2, l2_memory=L2Memory(size=4096), noc=noc)
