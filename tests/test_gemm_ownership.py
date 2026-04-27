from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorRange, TensorSlice
from MAPS.chips import magia_mesh
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.defs.gemm import GemmLayerOp


def test_build_gemm_tile_work_derives_required_operand_slices() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    tile = mesh.tile(1, 0)

    op = GemmLayerOp(
        x=Tensor(name="x", rank=2, dims=(8, 16), elem_bytes=2),
        w=Tensor(name="w", rank=2, dims=(16, 12), elem_bytes=2),
        y=Tensor(name="y", rank=2, dims=(8, 12), elem_bytes=2),
        output=Tensor(name="out", rank=2, dims=(8, 12), elem_bytes=2),
    )
    output_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )
    x_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )
    w_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )

    work = op.build_tile_work((x_layout, w_layout), (output_layout,), tile)

    assert work.output_slice == TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=4),
            TensorRange(start=6, length=6),
        ),
    )
    assert work.x_slice == TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=4),
            TensorRange(start=0, length=16),
        ),
    )
    assert work.w_slice == TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=16),
            TensorRange(start=6, length=6),
        ),
    )
    assert work.y_slice == work.output_slice
    assert work.l1_bytes == sum(ref.num_bytes for ref in work.input_slices + work.output_slices)
    assert work.fits_l1(tile)


def test_default_gemm_layouts_accept_logical_shape() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=6, height=1)
    op = GemmLayerOp(
        x=Tensor(name="x", rank=2, dims=(8, 16), elem_bytes=2),
        w=Tensor(name="w", rank=2, dims=(16, 12), elem_bytes=2),
        y=None,
        output=Tensor(name="out", rank=2, dims=(8, 12), elem_bytes=2),
    )

    input_layouts = op.default_input_layouts(
        submesh,
        logical_shape=(3, 2),
    )
    output_layouts = op.default_output_layouts(
        submesh,
        logical_shape=(3, 2),
    )

    assert all(layout.logical_width == 3 for layout in (*input_layouts, *output_layouts))
    assert all(layout.logical_height == 2 for layout in (*input_layouts, *output_layouts))
