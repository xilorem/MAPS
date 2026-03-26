from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorRange, TensorSlice
from MAPS.core.mesh import Mesh
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.builders.gemm_builder import build_gemm_tile_work
from MAPS.ops.gemm import GemmLayerOp


def test_build_gemm_tile_work_derives_required_operand_slices() -> None:
    mesh = Mesh(2, 2)
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
        microbatch_axis=None,
        num_microbatches=1,
    )
    x_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        microbatch_axis=None,
        num_microbatches=1,
    )
    w_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        microbatch_axis=None,
        num_microbatches=1,
    )

    work = build_gemm_tile_work(op, output_layout, x_layout, w_layout, tile, 0)

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
