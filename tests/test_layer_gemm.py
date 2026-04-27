from MAPS.core.graph import Node, OpKind
from MAPS.pipeline.layer import Layer
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.hw.chips import magia_mesh
from MAPS.pipeline.stage import Stage
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.defs.gemm import GemmPayload


def _make_layout(submesh: Submesh) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )

def test_stage_requires_at_least_one_layer() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)

    try:
        Stage(name="empty", submesh=submesh, layers=())
    except ValueError as exc:
        assert "at least one layer" in str(exc)
    else:
        raise AssertionError("expected empty stage construction to fail")


def test_stage_can_group_multiple_nodes() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    x = Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2)
    w0 = Tensor(name="w0", rank=2, dims=(8, 16), elem_bytes=2)
    y0 = Tensor(name="y0", rank=2, dims=(4, 16), elem_bytes=2)
    w1 = Tensor(name="w1", rank=2, dims=(16, 12), elem_bytes=2)
    y1 = Tensor(name="y1", rank=2, dims=(4, 12), elem_bytes=2)

    node0 = Node(
        name="gemm0",
        kind=OpKind.GEMM,
        inputs=(x, w0),
        outputs=(y0,),
        payload=GemmPayload(x=x, w=w0, y=None, output=y0),
    )
    node1 = Node(
        name="gemm1",
        kind=OpKind.GEMM,
        inputs=(y0, w1),
        outputs=(y1,),
        payload=GemmPayload(x=y0, w=w1, y=None, output=y1),
    )

    stage = Stage(
        name="stage0",
        submesh=submesh,
        layers=(
            Layer(node=node0),
            Layer(node=node1),
        ),
    )

    assert tuple(layer.node for layer in stage.layers) == (node0, node1)


def test_gemm_payload_accepts_valid_tensor_shapes() -> None:
    tensors = (
        Tensor(name="x", rank=3, dims=(2, 4, 8), elem_bytes=2),
        Tensor(name="w", rank=3, dims=(2, 8, 16), elem_bytes=2),
        Tensor(name="out", rank=3, dims=(2, 4, 16), elem_bytes=2),
        Tensor(name="y", rank=3, dims=(2, 4, 16), elem_bytes=2),
    )

    GemmPayload(
        x=tensors[0],
        w=tensors[1],
        y=tensors[3],
        output=tensors[2],
    )


def test_gemm_payload_rejects_incompatible_tensor_element_sizes() -> None:
    tensors = (
        Tensor(name="x", rank=3, dims=(2, 4, 8), elem_bytes=2),
        Tensor(name="w", rank=3, dims=(2, 8, 16), elem_bytes=4),
        Tensor(name="out", rank=3, dims=(2, 4, 16), elem_bytes=2),
    )

    try:
        GemmPayload(
            x=tensors[0],
            w=tensors[1],
            y=None,
            output=tensors[2],
        )
    except ValueError as exc:
        assert "element size" in str(exc)
    else:
        raise AssertionError("expected incompatible GEMM tensors to fail")


def test_gemm_payload_rejects_incompatible_k_dimension() -> None:
    tensors = (
        Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        Tensor(name="w", rank=2, dims=(7, 16), elem_bytes=2),
        Tensor(name="out", rank=2, dims=(4, 16), elem_bytes=2),
    )

    try:
        GemmPayload(
            x=tensors[0],
            w=tensors[1],
            y=None,
            output=tensors[2],
        )
    except ValueError as exc:
        assert "K dimension" in str(exc)
    else:
        raise AssertionError("expected incompatible GEMM K dimension to fail")
