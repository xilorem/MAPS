from MAPS.chips import magia_mesh
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.cost_models.exp_cost import ExpCostModel
from MAPS.ops.exp import ExpLayerOp


def _make_exp_op() -> ExpLayerOp:
    return ExpLayerOp(
        x=Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        output=Tensor(name="out", rank=2, dims=(4, 8), elem_bytes=2),
    )


def test_exp_tile_work_uses_output_slice_as_input_slice() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    op = _make_exp_op()
    output_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )

    tile_work = op.build_tile_work(
        input_layouts=(output_layout,),
        output_layouts=(output_layout,),
        tile=submesh.tiles[0],
    )

    assert tile_work.x_slice == tile_work.output_slice
    assert tuple((dim.start, dim.length) for dim in tile_work.output_slice.dims) == (
        (0, 4),
        (0, 4),
    )


def test_exp_cost_uses_exp_capable_device() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    op = _make_exp_op()
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )
    tile_work = op.build_tile_work(
        input_layouts=(layout,),
        output_layouts=(layout,),
        tile=submesh.tiles[0],
    )

    assert ExpCostModel().cost(tile_work, submesh.tiles[0]) == 32
