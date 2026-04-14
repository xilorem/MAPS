from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
)
from MAPS.chips import magia_mesh
from MAPS.core.ownership import _apply_layout_axis, tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor


def _format_slice_ranges(ranges: tuple[TensorRange, ...]) -> str:
    return ", ".join(
        f"axis{axis}=[{dim.start}:{dim.start + dim.length})"
        for axis, dim in enumerate(ranges)
    )


def test_tile_tensor_slice_shards_both_axes() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=3)
    tensor = Tensor(name="x", rank=3, dims=(8, 8, 12), elem_bytes=2)
    target_tile = mesh.tile(1, 2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=2),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
    )
    expected = (
        TensorRange(start=0, length=8),
        TensorRange(start=6, length=2),
        TensorRange(start=6, length=6),
    )

    result = tile_tensor_slice(
        tensor=tensor,
        layout=layout,
        tile=target_tile,
    )

    assert result.rank == 3
    assert result.dims == expected


def test_tile_tensor_slice_uses_logical_shape_not_physical_shape() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=6, height=1)
    tensor = Tensor(name="x", rank=2, dims=(6, 12), elem_bytes=2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
        logical_width=3,
        logical_height=2,
    )

    result = tile_tensor_slice(
        tensor=tensor,
        layout=layout,
        tile=mesh.tile(4, 0),
    )

    assert result.dims == (
        TensorRange(start=3, length=3),
        TensorRange(start=4, length=4),
    )
