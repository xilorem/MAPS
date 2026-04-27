from MAPS.arch import CoreDevice, DeviceKind, L1Memory, Tile, WorkKind
from MAPS.chips import magia_mesh
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.costs.conv_cost import ConvCostModel
from MAPS.ops.defs.conv import ConvPayload


def _make_conv_op() -> ConvPayload:
    return ConvPayload(
        x=Tensor(name="x", rank=4, dims=(1, 3, 5, 5), elem_bytes=2),
        w=Tensor(name="w", rank=4, dims=(8, 3, 3, 3), elem_bytes=2),
        b=Tensor(name="b", rank=1, dims=(8,), elem_bytes=2),
        output=Tensor(name="out", rank=4, dims=(1, 8, 3, 3), elem_bytes=2),
    )


def _replicated_layout(submesh: Submesh) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )


def test_conv_tile_work_uses_required_im2col_slices() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    op = _make_conv_op()
    layout = _replicated_layout(submesh)

    tile_work = op.build_tile_work(
        input_layouts=(layout, layout, layout),
        output_layouts=(layout,),
        tile=submesh.tiles[0],
    )

    assert tuple((dim.start, dim.length) for dim in tile_work.output_slice.dims) == (
        (0, 1),
        (0, 8),
        (0, 3),
        (0, 3),
    )
    assert tuple((dim.start, dim.length) for dim in tile_work.x_slice.dims) == (
        (0, 1),
        (0, 3),
        (0, 5),
        (0, 5),
    )
    assert tuple((dim.start, dim.length) for dim in tile_work.w_slice.dims) == (
        (0, 8),
        (0, 3),
        (0, 3),
        (0, 3),
    )
    assert tile_work.b_slice is not None
    assert tuple((dim.start, dim.length) for dim in tile_work.b_slice.dims) == ((0, 8),)
    assert tile_work.l1_bytes == sum(
        ref.num_bytes for ref in tile_work.input_slices + tile_work.output_slices
    )
    assert tile_work.fits_l1(submesh.tiles[0])


def test_conv_cost_uses_im2col_gemm_amount() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    op = _make_conv_op()
    layout = _replicated_layout(submesh)
    tile_work = op.build_tile_work(
        input_layouts=(layout, layout, layout),
        output_layouts=(layout,),
        tile=submesh.tiles[0],
    )

    scalar_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        devices=(
            CoreDevice(
                name="scalar_gemm",
                kind=DeviceKind.SCALAR,
                throughput={WorkKind.GEMM: 1},
            ),
        ),
        memory=L1Memory(size=4096, bandwidth=1),
    )
    redmule_tile = submesh.tiles[0]
    model = ConvCostModel()

    assert model.cost(tile_work, scalar_tile) == 1 * 8 * 3 * 3 * 3 * 3 * 3
    assert model.cost(tile_work, redmule_tile) < model.cost(tile_work, scalar_tile)
