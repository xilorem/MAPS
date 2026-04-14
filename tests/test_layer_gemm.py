from MAPS.core.graph import Node, OpKind
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.arch import L2Memory, Mesh
from MAPS.core.stage import (
    InputSource,
    InputSourceKind,
    Stage,
    StageInputBinding,
    StageOutputBinding,
)
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp


def _make_layout(submesh: Submesh) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )


def _make_input_binding(tensor_id: int) -> StageInputBinding:
    return StageInputBinding(
        tensor_id=tensor_id,
        source=InputSource(kind=InputSourceKind.EXTERNAL, external_base_addr=1),
    )


def test_stage_requires_at_least_one_node() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)

    try:
        Stage(name="empty", submesh=submesh, nodes=())
    except ValueError as exc:
        assert "at least one node" in str(exc)
    else:
        raise AssertionError("expected empty stage construction to fail")


def test_stage_can_group_multiple_nodes() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))
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
        payload=GemmLayerOp(x=x, w=w0, y=None, output=y0),
    )
    node1 = Node(
        name="gemm1",
        kind=OpKind.GEMM,
        inputs=(y0, w1),
        outputs=(y1,),
        payload=GemmLayerOp(x=y0, w=w1, y=None, output=y1),
    )

    stage = Stage(name="stage0", submesh=submesh, nodes=(node0, node1))

    assert len(stage.nodes) == 2


def test_gemm_payload_validates_binding_indices_and_tensor_shapes() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=3, dims=(2, 4, 8), elem_bytes=2),
        Tensor(name="w", rank=3, dims=(2, 8, 16), elem_bytes=2),
        Tensor(name="out", rank=3, dims=(2, 4, 16), elem_bytes=2),
        Tensor(name="y", rank=3, dims=(2, 4, 16), elem_bytes=2),
    )
    payload = GemmLayerOp(
        x=tensors[0],
        w=tensors[1],
        y=tensors[3],
        output=tensors[2],
    )

    payload.validate_tensors(
        inputs=(
            _make_input_binding(0),
            _make_input_binding(1),
            _make_input_binding(3),
        ),
        outputs=(StageOutputBinding(tensor_id=2, layout=layout),),
        tensors=tensors,
    )


def test_gemm_payload_rejects_missing_bound_tensor() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        Tensor(name="w", rank=2, dims=(8, 16), elem_bytes=2),
        Tensor(name="out", rank=2, dims=(4, 16), elem_bytes=2),
    )
    payload = GemmLayerOp(
        x=tensors[0],
        w=Tensor(name="missing_w", rank=2, dims=(8, 16), elem_bytes=2),
        y=None,
        output=tensors[2],
    )

    try:
        payload.validate_tensors(
            inputs=(
                _make_input_binding(0),
                _make_input_binding(1),
            ),
            outputs=(StageOutputBinding(tensor_id=2, layout=layout),),
            tensors=tensors,
        )
    except ValueError as exc:
        assert "GEMM W tensor is not present in stage inputs" in str(exc)
    else:
        raise AssertionError("expected missing GEMM tensor binding to fail")


def test_gemm_payload_rejects_incompatible_tensor_element_sizes() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=3, dims=(2, 4, 8), elem_bytes=2),
        Tensor(name="w", rank=3, dims=(2, 8, 16), elem_bytes=4),
        Tensor(name="out", rank=3, dims=(2, 4, 16), elem_bytes=2),
    )
    payload = GemmLayerOp(
        x=tensors[0],
        w=tensors[1],
        y=None,
        output=tensors[2],
    )

    try:
        payload.validate_tensors(
            inputs=(
                _make_input_binding(0),
                _make_input_binding(1),
            ),
            outputs=(StageOutputBinding(tensor_id=2, layout=layout),),
            tensors=tensors,
        )
    except ValueError as exc:
        assert "element size" in str(exc)
    else:
        raise AssertionError("expected incompatible GEMM tensors to fail")


def test_gemm_payload_rejects_incompatible_k_dimension() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        Tensor(name="w", rank=2, dims=(7, 16), elem_bytes=2),
        Tensor(name="out", rank=2, dims=(4, 16), elem_bytes=2),
    )
    payload = GemmLayerOp(
        x=tensors[0],
        w=tensors[1],
        y=None,
        output=tensors[2],
    )

    try:
        payload.validate_tensors(
            inputs=(
                _make_input_binding(0),
                _make_input_binding(1),
            ),
            outputs=(StageOutputBinding(tensor_id=2, layout=layout),),
            tensors=tensors,
        )
    except ValueError as exc:
        assert "K dimension" in str(exc)
    else:
        raise AssertionError("expected incompatible GEMM K dimension to fail")
