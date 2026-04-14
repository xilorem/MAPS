from MAPS.arch import L1Memory, L2Memory, Mesh, Tile
from MAPS.cost_models import TransportCostModel


def test_l2_transfer_cost_uses_l1_and_l2_bandwidth() -> None:
    slow_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=4),
    )
    fast_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=16),
    )
    slow_mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=4),
        tiles=(slow_tile,),
    )
    fast_mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=16),
        tiles=(fast_tile,),
    )

    assert TransportCostModel(mesh=fast_mesh).l1_to_l2(fast_tile, 64) < (
        TransportCostModel(mesh=slow_mesh).l1_to_l2(slow_tile, 64)
    )


def test_l2_transfer_cost_uses_nearest_l2_access_point_distance() -> None:
    mesh = Mesh(
        width=4,
        height=1,
        l2_memory=L2Memory(size=4096, access_points=((0, 0),), bandwidth=8),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=8)),
            Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=8)),
            Tile(tile_id=2, x=2, y=0, memory=L1Memory(size=4096, bandwidth=8)),
            Tile(tile_id=3, x=3, y=0, memory=L1Memory(size=4096, bandwidth=8)),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l2(mesh.tile(3, 0), 64) > model.l1_to_l2(mesh.tile(0, 0), 64)
    assert model.l2_to_l1(mesh.tile(3, 0), 64) > model.l2_to_l1(mesh.tile(0, 0), 64)


def test_l1_to_l1_transfer_cost_uses_tile_bandwidth_and_distance() -> None:
    src = Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=16))
    fast_dst = Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=16))
    slow_dst = Tile(tile_id=2, x=1, y=0, memory=L1Memory(size=4096, bandwidth=4))
    far_dst = Tile(tile_id=3, x=3, y=0, memory=L1Memory(size=4096, bandwidth=16))
    model = TransportCostModel()

    assert model.l1_to_l1(src, slow_dst, 64) > model.l1_to_l1(src, fast_dst, 64)
    assert model.l1_to_l1(src, far_dst, 64) > model.l1_to_l1(src, fast_dst, 64)
