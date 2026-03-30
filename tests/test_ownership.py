from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
)
from MAPS.arch import Mesh
from MAPS.core.ownership import _apply_layout_axis, tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor


def _format_slice_ranges(ranges: tuple[TensorRange, ...]) -> str:
    return ", ".join(
        f"axis{axis}=[{dim.start}:{dim.start + dim.length})"
        for axis, dim in enumerate(ranges)
    )


def test_tile_tensor_slice_microbatches_one_axis_and_shards_both_axes() -> None:
    mesh = Mesh(4, 4, l2_bytes=4096)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=3)
    tensor = Tensor(name="x", rank=3, dims=(8, 8, 12), elem_bytes=2)
    microbatch_idx = 1
    target_tile = mesh.tile(1, 2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=2),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        microbatch_axis=1,
        num_microbatches=2,
    )
    expected = (
        TensorRange(start=0, length=8),
        TensorRange(start=7, length=1),
        TensorRange(start=6, length=6),
    )

    print(
        "\n=== Ownership Test: microbatch on one sharded axis, "
        "and shard on both mesh axes ==="
    )
    print(f"mesh: {mesh.x_size}x{mesh.y_size}")
    print(f"submesh: origin=({submesh.x0}, {submesh.y0}) size={submesh.width}x{submesh.height}")
    print(f"tensor: name={tensor.name}, rank={tensor.rank}, dims={tensor.dims}")
    print(
        "layout: "
        f"microbatch_axis={layout.microbatch_axis}, "
        f"num_microbatches={layout.num_microbatches}, "
        f"mesh_y=SHARD(axis={layout.mesh_y.tensor_axis}), "
        f"mesh_x=SHARD(axis={layout.mesh_x.tensor_axis})"
    )
    print(f"selected microbatch: {microbatch_idx}")
    print(
        f"expected slice for tile {target_tile.tile_id} at ({target_tile.x}, {target_tile.y}): "
        f"{_format_slice_ranges(expected)}"
    )
    print("computed slices for every tile:")

    for y in range(mesh.y_size):
        for x in range(mesh.x_size):
            tile = mesh.tile(x, y)
            if submesh.contains_tile_id(tile.tile_id):
                tile_slice = tile_tensor_slice(
                    tensor=tensor,
                    layout=layout,
                    tile=tile,
                    microbatch_idx=microbatch_idx,
                )
                print(
                    f"  tile {tile.tile_id:2d} ({tile.x}, {tile.y}) -> "
                    f"{_format_slice_ranges(tile_slice.dims)}"
                )
            else:
                print(f"  tile {tile.tile_id:2d} ({tile.x}, {tile.y}) -> outside submesh")

    result = tile_tensor_slice(
        tensor=tensor,
        layout=layout,
        tile=target_tile,
        microbatch_idx=microbatch_idx,
    )

    assert result.rank == 3
    assert result.dims == expected
    print("result: PASS")
