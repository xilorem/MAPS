from MAPS.core.layer import (
    InputSource,
    InputSourceKind,
    Layer,
    LayerInputBinding,
    LayerOpKind,
    LayerOutputBinding,
)
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.core.mesh import Mesh
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp


def _make_layout(submesh: Submesh) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        microbatch_axis=None,
        num_microbatches=1,
    )


def _make_input_binding(tensor_id: int) -> LayerInputBinding:
    return LayerInputBinding(
        tensor_id=tensor_id,
        source=InputSource(kind=InputSourceKind.EXTERNAL, external_base_addr=1),
    )


def test_gemm_layer_requires_gemm_payload() -> None:
    mesh = Mesh(2, 2)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)

    try:
        Layer(
            name="gemm",
            submesh=submesh,
            kind=LayerOpKind.GEMM,
            inputs=(),
            outputs=(),
            payload=None,
        )
    except ValueError as exc:
        assert "GEMM layers require a GemmLayerOp payload" in str(exc)
    else:
        raise AssertionError("expected GEMM layer construction to fail")


def test_non_gemm_layer_rejects_gemm_payload() -> None:
    mesh = Mesh(2, 2)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)

    try:
        Layer(
            name="eltwise",
            submesh=submesh,
            kind=LayerOpKind.ELEMENTWISE,
            payload=GemmLayerOp(
                x=Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
                w=Tensor(name="w", rank=2, dims=(8, 16), elem_bytes=2),
                y=None,
                output=Tensor(name="out", rank=2, dims=(4, 16), elem_bytes=2),
            ),
        )
    except ValueError as exc:
        assert "GemmLayerOp payloads are only valid for GEMM layers" in str(exc)
    else:
        raise AssertionError("expected non-GEMM layer construction to fail")


def test_gemm_layer_validates_binding_indices_and_tensor_shapes() -> None:
    mesh = Mesh(2, 2)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=3, dims=(2, 4, 8), elem_bytes=2),
        Tensor(name="w", rank=3, dims=(2, 8, 16), elem_bytes=2),
        Tensor(name="out", rank=3, dims=(2, 4, 16), elem_bytes=2),
        Tensor(name="y", rank=3, dims=(2, 4, 16), elem_bytes=2),
    )

    layer = Layer(
        name="gemm",
        submesh=submesh,
        kind=LayerOpKind.GEMM,
        inputs=(
            _make_input_binding(0),
            _make_input_binding(1),
            _make_input_binding(3),
        ),
        outputs=(LayerOutputBinding(tensor_id=2, layout=layout),),
        payload=GemmLayerOp(
            x=tensors[0],
            w=tensors[1],
            y=tensors[3],
            output=tensors[2],
        ),
    )

    layer.validate_tensors(tensors)


def test_gemm_layer_rejects_missing_bound_tensor() -> None:
    mesh = Mesh(2, 2)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        Tensor(name="w", rank=2, dims=(8, 16), elem_bytes=2),
        Tensor(name="out", rank=2, dims=(4, 16), elem_bytes=2),
    )

    layer = Layer(
        name="gemm_bad",
        submesh=submesh,
        kind=LayerOpKind.GEMM,
        inputs=(
            _make_input_binding(0),
            _make_input_binding(1),
        ),
        outputs=(LayerOutputBinding(tensor_id=2, layout=layout),),
        payload=GemmLayerOp(
            x=tensors[0],
            w=Tensor(name="missing_w", rank=2, dims=(8, 16), elem_bytes=2),
            y=None,
            output=tensors[2],
        ),
    )

    try:
        layer.validate_tensors(tensors)
    except ValueError as exc:
        assert "GEMM W tensor is not present in layer inputs" in str(exc)
    else:
        raise AssertionError("expected missing GEMM tensor binding to fail")


def test_gemm_layer_rejects_incompatible_tensor_element_sizes() -> None:
    mesh = Mesh(2, 2)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=3, dims=(2, 4, 8), elem_bytes=2),
        Tensor(name="w", rank=3, dims=(2, 8, 16), elem_bytes=4),
        Tensor(name="out", rank=3, dims=(2, 4, 16), elem_bytes=2),
    )

    layer = Layer(
        name="gemm_bad",
        submesh=submesh,
        kind=LayerOpKind.GEMM,
        inputs=(
            _make_input_binding(0),
            _make_input_binding(1),
        ),
        outputs=(LayerOutputBinding(tensor_id=2, layout=layout),),
        payload=GemmLayerOp(
            x=tensors[0],
            w=tensors[1],
            y=None,
            output=tensors[2],
        ),
    )

    try:
        layer.validate_tensors(tensors)
    except ValueError as exc:
        assert "element size" in str(exc)
    else:
        raise AssertionError("expected incompatible GEMM tensors to fail")


def test_gemm_layer_rejects_incompatible_k_dimension() -> None:
    mesh = Mesh(2, 2)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        Tensor(name="w", rank=2, dims=(7, 16), elem_bytes=2),
        Tensor(name="out", rank=2, dims=(4, 16), elem_bytes=2),
    )

    layer = Layer(
        name="gemm_bad_k",
        submesh=submesh,
        kind=LayerOpKind.GEMM,
        inputs=(
            _make_input_binding(0),
            _make_input_binding(1),
        ),
        outputs=(LayerOutputBinding(tensor_id=2, layout=layout),),
        payload=GemmLayerOp(
            x=tensors[0],
            w=tensors[1],
            y=None,
            output=tensors[2],
        ),
    )

    try:
        layer.validate_tensors(tensors)
    except ValueError as exc:
        assert "K dimension" in str(exc)
    else:
        raise AssertionError("expected incompatible GEMM K dimension to fail")
